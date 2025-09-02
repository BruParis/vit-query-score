model = dict(
    type="VitWrapper",
    backbone=dict(
        type="VisionTransformer",
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    # cls_head=dict(
    #     type="ClassificationHead",
    #     num_classes=400,
    #     in_channels=768,
    # ),
)
