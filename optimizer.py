from torch.optim import SGD, Adam


def get_optimizer(optimizer_name, lr, model):
    model_params = (p for p in model.parameters() if p.requires_grad)
    print(f'Using {optimizer_name} optimizer')
    if optimizer_name == 'sgd':
        return SGD(
            model_params,
            lr=lr,
            weight_decay=0,
            momentum=0.9,
            nesterov=True,
        )
    elif optimizer_name == 'adam':
        return Adam(model_params, lr=lr)
    else:
        raise Exception(f'Unknown optimizer: {optimizer_name}')
