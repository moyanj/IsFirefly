import torch
import os

checkpoint = torch.load("models/IsACG-v1-99.06%.pt")

torch.save(checkpoint, "models/IsACG_v1_99.06%_raw.pt")

os.remove("models/IsACG-v1-99.06%.pt")

del checkpoint["optimizer_state"]
del checkpoint["scheduler_state"]
del checkpoint["losses"]
del checkpoint["val_loss"]
del checkpoint["val_accuracy"]
del checkpoint["best_accuracy"]

torch.save(checkpoint, "models/IsACG_v1_99.06%.pt")
