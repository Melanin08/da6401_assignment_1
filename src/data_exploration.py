import wandb
from utils.data_loader import load_data

wandb.init(project="da6401-assignment")

X_train, y_train, _, _ = load_data("mnist")

table = wandb.Table(columns=["image","label"])

count = {i:0 for i in range(10)}

for img,label in zip(X_train,y_train):
    if count[label] < 5:
        table.add_data(wandb.Image(img.reshape(28,28)), label)
        count[label]+=1
        
    if sum(count.values()) == 50:
        break

wandb.log({"sample_images": table})

wandb.finish()