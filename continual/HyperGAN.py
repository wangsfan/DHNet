import copy

import torch
from timm.models.layers import trunc_normal_
from torch import nn

from continual.cnn import resnet18
import continual.utils as cutils
from continual.Hyper_GAN_vit import HyperBlock, Generator, Hyper_GAN_Vit


class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""

    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n


class HyperGAN(nn.Module):
    def __init__(
            self,
            transformer,
            nb_classes,
            individual_classifier='',
            head_div=False,
            head_div_mode=['tr', 'ft'],
            resnet=False
    ):
        super().__init__()
        self.nb_classes = nb_classes

        self.individual_classifier = individual_classifier
        self.use_head_div = head_div
        self.head_div_mode = head_div_mode
        self.head_div = None
        self.in_finetuning = False

        self.use_resnet = resnet

        self.nb_classes_per_task = [nb_classes]

        if self.use_resnet:
            print('ResNet18 backbone for ens')
            self.backbone = resnet18()
            self.backbone.head = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 504, kernel_size=1),
                nn.BatchNorm2d(504),
                nn.ReLU(inplace=True)
            )
            self.backbone.avgpool = nn.Identity()
            self.backbone.layer4 = nn.Identity()
            # self.backbone.layer4 = self.backbone._make_layer_nodown(
            #    256, 512, 2, stride=1, dilation=2
            # )
            self.backbone = self.backbone.cuda()
            self.backbone.embed_dim = 504
            self.embed_dim = 504

            self.decoder = nn.ModuleList([
                HyperBlock(
                    dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm
                ).cuda()
            ])
            self.hyper_len = len(self.decoder)
            z = nn.Parameter(torch.fmod(torch.randn(self.embed_dim, self.hyper_len).cuda(), 2))
            trunc_normal_(z, std=.02)
            self.hyper_conditions = nn.ParameterList([z])
            self.hyperFC = Generator(100, 4, 64)


        else:
            self.embed_dim = transformer.embed_dim
            self.patch_embed = transformer.patch_embed
            self.pos_embed = transformer.pos_embed
            self.pos_drop = transformer.pos_drop
            self.sabs = transformer.blocks[:transformer.local_up_to_layer]
            self.decoder = transformer.blocks[transformer.local_up_to_layer:]
            self.hyper_len = len(self.decoder)
            self.z = nn.Parameter(torch.fmod(torch.randn(self.hyper_len, 16, 100).cuda(), 2))
            trunc_normal_(self.z, std=.02)
            self.hyper_conditions = nn.ParameterList([self.z])
            self.hyperFC = transformer.HyperFC
            self.z_dim = self.z.shape

            # ***********
            # self.hyperMLP = transformer.Hyper_mlp

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, self.embed_dim))
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.hyper_conditions), sum(self.nb_classes_per_task)
            ).cuda()

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # ************************** new hyper Decoder *************************
        new_condition = nn.Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2))
        trunc_normal_(new_condition, std=.02)
        self.hyper_conditions.append(new_condition)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            self.head_div = ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.hyper_conditions), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.
        """
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.hyper_conditions) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.hyper_conditions) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_conditions':
                cutils.freeze_parameters(self.hyper_conditions[:-1], requires_grad=requires_grad)
            elif name == 'hyper_conditions':
                cutils.freeze_parameters(self.hyper_conditions, requires_grad=requires_grad)
            elif name == 'sab':
                if self.use_resnet:
                    self.backbone.eval()
                    cutils.freeze_parameters(self.backbone, requires_grad=requires_grad)
                else:
                    self.sabs.eval()
                    cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.pos_embed, requires_grad=requires_grad)
                    cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'decoder':
                self.decoder.eval()
                cutils.freeze_parameters(self.decoder, requires_grad=requires_grad)


            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_conditions': self.hyper_conditions[:-1],
            'hyper_conditions': self.hyper_conditions.parameters(),
            'new_conditions': [self.hyper_conditions[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'decoder': self.decoder.parameters(),
            'hyper_networks': self.hyperFC.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                if self.individual_classifier else \
                self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""

        log = {}
        # Compute mean distance between hyper-networks' conditions
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.hyper_conditions)):
                for j in range(i + 1, len(self.hyper_conditions)):
                    dist = torch.norm(self.hyper_conditions[i] - self.hyper_conditions[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        log['hyper_conditions_len'] = len(self.hyper_conditions)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        if self.use_resnet:
            x, self.feats = self.backbone.forward_tokens(x)
        else:
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            self.feats = []
            for blk in self.sabs:
                x, attn, v = blk(x)
                self.feats.append(x)
            self.feats.pop(-1)

        outs = []
        attentions = []
        cls = x

        for condition in self.hyper_conditions:
            if self.decoder is not None:
                for i, blk in enumerate(self.decoder):
                    # print(condition.shape)
                    cls, attn, weight = blk(x, condition[i, :, :], self.hyperFC)  # ignore weight distillation temporary!

                attentions.append(attn)
            outs.append(self.avg_pooling(cls).squeeze())

        self._class_tokens = outs

        return outs, outs[-1], attentions

    def forward_classifier(self, outs, last_out):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param outs: A list of all task outs embeddings.
        :param last_out: The ultimate task out embedding from the latest task.
        """
        logits_div = None

        if self.individual_classifier != '':
            logits = []

            for i, head in enumerate(self.head):
                if self.individual_classifier in ('1-n', '1-1'):
                    logits.append(head(outs[i]))
                else:  # n-1, n-n
                    logits.append(head(torch.cat(outs[:i + 1], dim=1)))

            if self.individual_classifier in ('1-1', 'n-1'):
                logits = torch.cat(logits, dim=1)
            else:  # 1-n, n-n
                final_logits = torch.zeros_like(logits[-1])
                for i in range(len(logits)):
                    final_logits[:, :logits[i].shape[1]] += logits[i]

                for i, c in enumerate(self.nb_classes_per_task):
                    final_logits[:, :c] /= len(self.nb_classes_per_task) - i

                logits = final_logits
        elif isinstance(outs, torch.Tensor):
            logits = self.head(outs)
        else:
            logits = self.head(torch.cat(outs, dim=1))

        if self.head_div is not None and eval_training_finetuning(self.head_div_mode, self.in_finetuning):
            logits_div = self.head_div(last_out)  # only last out

        return {
            'logits': logits,
            'div': logits_div,
            'outs': outs
        }

    def forward(self, x):
        outs, last_out, _ = self.forward_features(x)
        return self.forward_classifier(outs, last_out)


def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False

