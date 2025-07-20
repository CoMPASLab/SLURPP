from .slurpp_trainer import SlurppTrainer


trainer_cls_name_dict = {
    "SlurppTrainer": SlurppTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
