
def train(model, data):
    steps_per_epoch = data.traingenerator.n//data.traingenerator.batch_size
    val_steps = data.validationgenerator.n//data.validationgenerator.batch_size

    return model.model.fit(
        data.traingenerator,
        steps_per_epoch=steps_per_epoch,
        epochs=model.nbepochs,
        validation_data=data.validationgenerator,
        validation_steps=val_steps)