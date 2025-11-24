intializators = ['random_normal','he_normal','glorot_uniform']
for init in intializators:
    models = models.Sequential([
        layers.Dense(64, kernel_initializer=init, input_shape=(100,)),
        layers.Dense(10, activation='softmax')
    ])
    w,b = model.layers[0].get_weights()
    print(f'Initializer:{init}')
    print(f'mean: {np.mean(w)}, std:{np.std(w)}')
    print(f'min: {np.min(w)}, max:{np.max(w)}')