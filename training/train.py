from modules import training_module as training

dataset = training.dataset()
print(dataset.train_frequency)
print(dataset.val_frequency)

# target_aus = {'25','12','7','14','24','11','10','20','4','17','26','6','1','2','15','9','20'}
# dataset.filterAus(target_aus)
dataset.makeGenerators()

# print(dataset.train_frequency)
val_generator = dataset.val_generator
train_generator = dataset.train_generator

train = training.train(train_generator,val_generator)
