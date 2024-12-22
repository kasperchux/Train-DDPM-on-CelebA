from diffusers import UNet2DModel
from configuration import config

model = UNet2DModel(
    sample_size=config.image_size, # размер изображения
    in_channels=3, # кол-во входных каналов (3 потому что формат изображений - RGB)
    out_channels=3, # кол-во выходных каналов
    layers_per_block=2, # кол-во ResNet слоев в каждом UNet блоке
    block_out_channels=(128, 128, 256, 256, 512, 512), # кол-ва выходных каналов для каждого UNet блока
    down_block_types=( # здесь описываем блоки которые используем
        "DownBlock2D", # здесь для понижения
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D", # используем блок с механизмом внимания
        "DownBlock2D",
    ),
    up_block_types=( # здесь блоки для повышения
        "UpBlock2D",
        "AttnUpBlock2D", # снова блок с механизмом внимания
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)