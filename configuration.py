from dataclasses import dataclass

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64  # размер изображений
    train_batch_size = 64 # размер батча в обучении
    eval_batch_size = 16  # размер батча при тестировании
    num_epochs = 100 # количество эпох 
    gradient_accumulation_steps = 1 # 
    learning_rate = 1e-4 # скорость обучения
    lr_warmup_steps = 500 
    save_image_epochs = 5 # раз во сколько эпох мы..
    # .. будем во время обучения запускать проверочную генерацию изображений
    save_model_epochs = 10 # раз в сколько эпох мы будем сохранять модель
    mixed_precision = "fp16" # точность
    output_dir = "CelebA-faces-ddpm" # папка для сохранения
    overwrite_output_dir = True # переписывать ли предыдущую модель
    seed = 0


config = TrainingConfig()