current_dataset = 'Hermiston'
current_model = '_1'
current = current_dataset + current_model

# 0.Super parameter
train_set_num = [0.002, 0.1]   # label, unlabel
bs_number = [64, 256]
lr = 1e-1
epoch_number = [110, 1]
lr_step = [30, 60, 120]

# 1. data
phase = ['trl', 'tru', 'test', 'gt']
patch_size = 7
data = dict(
    current_dataset=current_dataset,
    train_set_num=train_set_num,
    patch_size=patch_size,

    trl_data=dict(
        phase=phase[0]
    ),
    tru_data=dict(
        phase=phase[1]
    ),
    test_data=dict(
        phase=phase[2]
    ),
)

# 2. model
model = {'Hermiston': [154, 1024, 2]}

# 3. train
train = dict(
    optimizer=dict(
        # typename='Adm',
        typename='SGD',
        lr=lr,
        momentum=0.9,
        weight_decay=1e-3
    ),
    train_UL_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=12,
        epoch=epoch_number[0],
        batch_size_l=bs_number[0],
        batch_size_u=bs_number[1],
        lr=lr,
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=lr_step,
        save_folder='./weights/' + current_dataset + '/',
        save_name=current,
        reuse_model=False,
        reuse_file='./weights/' + current + '_Final.pth',
    )
)

# 4. test
test = dict(
    current_dataset=current_dataset,
    batch_size=100,
    gpu_train=True,
    gpu_num=1,
    workers_num=8,
    model_weights='./weights/' + current_dataset + '/' + current + '_Final.pth',
    save_name=current,
    save_folder='./result' + '/' + current_dataset
)
