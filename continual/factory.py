import torch

from continual import samplers, vit,Hyper_GAN_vit, HyperGAN


def get_backbone(args):
    print(f"Creating model: {args.model}")
    if args.model == 'vit':
        model = vit.VisionTransformer(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads
        )

    elif args.model == 'hyper_gan_vit':
        model = Hyper_GAN_vit.Hyper_GAN_Vit(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            local_up_to_layer=args.local_up_to_layer,
            locality_strength=args.locality_strength,
            norm_layer=args.norm,
        )
    else:
        raise NotImplementedError(f'Unknown backbone {args.model}')

    return model


def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = samplers.get_sampler(dataset_train, dataset_val, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None if (finetuning and args.ft_no_sampling) else sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(sampler_train) > args.batch_size,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, batch_size=None, drop_last=True):
    batch_size = batch_size or args.batch_size

    sampler_train = samplers.get_train_sampler(dataset_train, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )

    return loader_train


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return self.get()


def update_hyper_gan(model_without_ddp, task_id, args):
    if task_id == 0:
        print(f'Creating hyperGAN!')
        model_without_ddp = HyperGAN.HyperGAN(
            model_without_ddp,
            nb_classes=args.initial_increment,
            individual_classifier=args.ind_clf,
            head_div=args.head_div > 0.,
            head_div_mode=args.head_div_mode,
            resnet=args.resnet
        )
    else:
        print(f'Updating ensemble, new embed dim {model_without_ddp.embed_dim}.')
        model_without_ddp.add_model(args.increment)

    return model_without_ddp



