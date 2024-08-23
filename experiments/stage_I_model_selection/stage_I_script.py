import timm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_models():
    FILE_NAME = 'model_descriptions.csv'
    with open(FILE_NAME, "a+") as file:
        file.write("name,parameters\n")
    
    #resnets = ['resnet10t', 'resnet14t', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet26t', 'resnet32ts', \
        # 'resnet33ts', 'resnet34', 'resnet34d', 'resnet50', 'resnet50c', 'resnet50d', 'resnet50s', 'resnet50t', \
        #     'resnet101', 'resnet101d', 'resnet101d']
    
    models = timm.list_models()
    
    for model_name in models:
        model = timm.create_model(model_name, num_classes=10)
        parameters_count = count_parameters(model)
        with open(FILE_NAME, "a+") as file:
            file.write(f"{model_name},{parameters_count}\n")
            

if __name__ == "__main__":
    list_models()