import numpy as np
import tensorflow as tf
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
import random

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


"""
initializer in this paper
"""
initializer_local = tf.keras.initializers.GlorotUniform(42)

"""
Generate spatial-space coefficient
"""


def distance_bound(data):
    # r z t
    dist = 0
    data_one_axis = data[:, 0:1]
    data_norm = (data_one_axis - np.min(data_one_axis)) / (np.max(data_one_axis) - np.min(data_one_axis))
    dist = dist + 4 * 10 * (data_norm - 0.5) ** 2 + 1

    data_one_axis = data[:, 1:2]
    data_norm = (data_one_axis - np.min(data_one_axis)) / (np.max(data_one_axis) - np.min(data_one_axis))
    dist = dist + 4 * 10 * (data_norm - 0.5) ** 2 + 1

    data_one_axis = data[:, 2:3]
    data_norm = (data_one_axis - np.min(data_one_axis)) / (np.max(data_one_axis) - np.min(data_one_axis))
    dist_t = 4 * 10 * (data_norm - 0.5) ** 2 + 1

    dist_t[np.where(
        data_one_axis[:, 0:1] > (np.max(data_one_axis) - np.min(data_one_axis)) / 2 + np.min(data_one_axis))] = 1
    weight1 = dist + dist_t
    # r z t
    dist = 0
    data_one_axis = data[:, 1:2]
    data_norm = (data_one_axis - np.min(data_one_axis)) / (np.max(data_one_axis) - np.min(data_one_axis))
    dist = dist + 4 * 10 * (data_norm - 0.5) ** 2 + 1

    data_one_axis = data[:, 2:3]
    data_norm = (data_one_axis - np.min(data_one_axis)) / (np.max(data_one_axis) - np.min(data_one_axis))
    dist_t = 4 * 10 * (data_norm - 0.5) ** 2 + 1

    dist_t[np.where(
        data_one_axis[:, 0:1] > (np.max(data_one_axis) - np.min(data_one_axis)) / 2 + np.min(data_one_axis))] = 1
    weight2 = dist + dist_t

    return weight1, weight2


"""
Generate boundary initial and pde points
"""


def datagen(rpoints, zpoints, tpoints):
    pde_r = np.linspace(0, 1, rpoints)
    pde_z = np.linspace(0, 1, zpoints)
    pde_t = np.linspace(0, 1, tpoints)
    pdepoints = np.array(np.meshgrid(pde_r, pde_z, pde_t), dtype=np.float32).T.reshape(-1, 3)
    initialpoints = int(rpoints * zpoints * tpoints)

    initial = np.linspace([1 / rpoints, 1 / zpoints, 0], [1 - 1 / rpoints, 1 - 1 / zpoints, 0], initialpoints)

    boundarypoints = int(rpoints * zpoints * tpoints / 4)
    boundr = np.linspace(0, 1, 2)
    boundz = np.linspace(0, 1, 2)
    boundt = np.linspace(0, 1, boundarypoints)
    bound = np.array(np.meshgrid(boundr, boundz, boundt), dtype=np.float32).T.reshape(-1, 3)

    weight1, weight2 = distance_bound(pdepoints)

    pde_y = np.zeros([pdepoints.shape[0]])
    bound_y = np.zeros([bound.shape[0]])
    init_y = np.zeros([initial.shape[0]])

    return {'pde': pdepoints, 'init': initial, 'bound': bound}, {'output_1': pde_y, 'output_2': init_y,
                                                                 'output_3': bound_y}, \
        weight1, weight2


"""
Two-stage training process(callbacks version)
"""
pretrainflag = 0


class Twostage(tf.keras.callbacks.Callback):
    def __init__(self, tol, rate, balancep):
        super().__init__()
        self.tol = tol  # [tol_T, tol_Ur, tol_Uz]
        self.rate = rate
        self.balancep = balancep

    def on_epoch_end(self, epoch, logs={}):
        global pretrainflag
        lossweight = self.model.compiled_loss._loss_weights  # getting loss weight at present in the model
        constrainloss = logs.get('output_2_loss') + logs.get('output_3_loss')  # calculate constrained loss
        pdeloss = logs.get('output_1_loss')  # get pde loss
        dispcons = logs.get('output_2_loss') * lossweight[1] + logs.get('output_3_loss') * lossweight[
            2]  # calculate the constrained loss after balance
        disppde = logs.get('output_1_loss') * lossweight[0].numpy()  # calculate pde loss after balance
        loss = logs.get('loss')  # total loss for disp

        if pretrainflag == 0:  # pretrain flag is a global variable
            if constrainloss > self.tol:
                newkfactor = lossweight[
                                 0] * 0.999  # if the total constrained loss larger than the tolerant, then keep balancing
                self.model.compiled_loss._loss_weights[0] = tf.convert_to_tensor(newkfactor,
                                                                                 dtype=tf.float32)  # process by repalce the loss weight
            elif epoch >= 1000:
                pretrainflag = 1  # End pre training in advance
            else:
                pretrainflag = 1  # Meet the conditions and end the pre training
        else:
            # Formal training
            if (constrainloss < self.tol) & ((pdeloss * lossweight[0]) < (self.rate * self.balancep)):
                newkfactor = lossweight[0] * 1.05
                self.model.compiled_loss._loss_weights[0] = tf.convert_to_tensor(newkfactor, dtype=tf.float32)
        kfactor_history.append(lossweight[0].numpy())
        if epoch % 100 == 0:  # disp
            print('Iteration:', epoch, 'Loss:', loss, 'PDE:', disppde, 'PDEreal', pdeloss, 'Constrainedreal:',
                  constrainloss, 'Constrained:', dispcons, 'kfactor:', lossweight[0].numpy(), 'lr:',
                  self.model.optimizer.lr.numpy())


"""
Coupled model for heat conduction
"""


class Coupled_T(tf.keras.Model):
    def __init__(self):
        super(Coupled_T, self).__init__()
        self.layer_heat = myblock(units=NN_FILTER, stacksize=NN_DEEP)
        self.output_heat = tf.keras.layers.Dense(1, kernel_initializer=initializer_local,
                                                 bias_initializer=initializer_local, name='out_T')

    def call(self, input_tensor, **kwargs):
        pde_cord = input_tensor['pde']
        with tf.GradientTape(persistent=True) as second:
            second.watch(pde_cord)
            with tf.GradientTape(persistent=True) as first:
                first.watch(pde_cord)
                heat = self.layer_heat(pde_cord)
                out_T = self.output_heat(heat)
                out_T_TO = out_T * T_rate - T0
            dTd = first.gradient(out_T, pde_cord)
            dTdr = dTd[:, 0:1] / rmax
            dTdz = dTd[:, 1:2] / zmax
            dTdt = dTd[:, 2:3] / tmax
            dT_T0dz = first.gradient(out_T_TO, pde_cord)[:, 1:2] / zmax

        dTdzz = second.gradient(dTdz, pde_cord)[:, 1:2] / zmax
        dTdrr = second.gradient(dTdr, pde_cord)[:, 0:1] / rmax

        del first, second
        dUzdzt_T = input_tensor['coupled']

        initial_cord = input_tensor['init']
        init = self.layer_heat(initial_cord)
        out_init = self.output_heat(init)

        bound_cord = input_tensor['bound']
        with tf.GradientTape() as first:
            first.watch(bound_cord)
            bn = self.layer_heat(bound_cord)
            out_bn = self.output_heat(bn)
        dTdr_bn = first.gradient(out_bn, bound_cord)[:, 0:1] / rmax

        Bottom = 0.5 * (1.0 - tf.math.sign(bound_cord[:, 1:2] - 0 - 1e-15))
        Top = 0.5 * (1.0 + tf.math.sign(bound_cord[:, 1:2] - 1 + 1e-15))
        Left = 0.5 * (1.0 - tf.math.sign(bound_cord[:, 0:1] - 0 - 1e-15))
        Right = 0.5 * (1.0 + tf.math.sign(bound_cord[:, 0:1] - 1 + 1e-15))

        cons_bound = tf.abs(out_bn - 1700. / T_rate) * Bottom + tf.abs(out_bn - 500. / T_rate) * Top + tf.abs(
            dTdr_bn - 0) * Left + tf.abs(
            dTdr_bn - 0) * Right
        cons_init = tf.abs(out_init - 500. / T_rate)

        pde_T = ((dTdzz + dTdrr) - p * Cp * dTdt / K - (T0 / 1700) * (beta3 * dUzdzt_T * Uz_rate) / K)

        return pde_T, cons_init, cons_bound, dT_T0dz


"""
Coupled model for stress conduction
"""


class Coupled_Uz(tf.keras.Model):
    def __init__(self):
        super(Coupled_Uz, self).__init__()
        self.layer_Uz = myblock(units=NN_FILTER, stacksize=NN_DEEP)
        self.output_Uz = tf.keras.layers.Dense(1, kernel_initializer=initializer_local,
                                               bias_initializer=initializer_local, name='out_Uz')

    def call(self, input_tensor, **kwargs):
        # 输入数据
        pde_cord = input_tensor['pde']
        with tf.GradientTape(persistent=True) as second:
            second.watch(pde_cord)
            with tf.GradientTape(persistent=True) as first:
                first.watch(pde_cord)
                stress_uz = self.layer_Uz(pde_cord)
                out_Uz = self.output_Uz(stress_uz)
            dUzd = first.gradient(out_Uz, pde_cord)
            dUzdr = dUzd[:, 0:1] / rmax
            dUzdz = dUzd[:, 1:2] / zmax
            dUzdt = dUzd[:, 2:3] / tmax
        dUzdrr = second.gradient(dUzdr, pde_cord)[:, 0:1] / rmax
        dUzdzz = second.gradient(dUzdz, pde_cord)[:, 1:2] / zmax
        dUzdtt = second.gradient(dUzdt, pde_cord)[:, 2:3] / tmax
        dUzdzt = second.gradient(dUzdz, pde_cord)[:, 2:3] / tmax
        dUzdrz = second.gradient(dUzdr, pde_cord)[:, 1:2] / zmax
        del first, second

        dT_T0dz_Uz = input_tensor['coupled']

        initial_cord = input_tensor['init']
        init = self.layer_Uz(initial_cord)
        out_init = self.output_Uz(init)

        bound_cord = input_tensor['bound']
        bn = self.layer_Uz(bound_cord)
        out_bn = self.output_Uz(bn)

        Bottom = 0.5 * (1.0 - tf.math.sign(bound_cord[:, 1:2] - 0 - 1e-15))
        Top = 0.5 * (1.0 + tf.math.sign(bound_cord[:, 1:2] - 1 + 1e-15))

        cons_bound = tf.abs(out_bn - 0) * Bottom + tf.abs(out_bn - 0) * Top
        cons_init = tf.abs(out_init - 0)

        pde_Uz = p * dUzdtt / C11 * Uz_rate - dUzdzz * Uz_rate - (C441 / 2) * dUzdrr * Uz_rate / C11 + \
                 beta3 * dT_T0dz_Uz / C11 + (C12 + (C441 / 2)) / C11 * dUzdrz * Uz_rate
        return pde_Uz, cons_init, cons_bound, dUzdzt


"""
block model
"""


class myblock(tf.keras.layers.Layer):
    def __init__(self, units=30, stacksize=10):
        super(myblock, self).__init__()
        self.stacksize = stacksize
        self.dense1 = [
            tf.keras.layers.Dense(units, kernel_initializer=initializer_local, bias_initializer=initializer_local)
            for l in range(stacksize)
        ]

    def call(self, inputs, **kwargs):
        out = inputs

        for i in range(self.stacksize):
            out = self.dense1[i](out)
            out = tf.nn.tanh(out)

        return out


"""
loss function
"""


def mse(x, sampleweight):
    return tf.reduce_mean(tf.square(x) * sampleweight)


def mae(x):
    return tf.reduce_mean(tf.abs(x))


"""
adaptive learning rate
"""
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='output_1_loss',
                                                 factor=0.9, patience=100,
                                                 verbose=0, mode='auto',
                                                 min_delta=0.0001, cooldown=0, min_lr=1e-5)

"""
get coupled term and print result
"""


def printresult(model, bcp, tol, x, coupledin, weight):
    x.update({'coupled': coupledin})
    pde, cons_init, cons_bn, coupledout = model(x)
    C = mae(cons_init).numpy() + mae(cons_bn).numpy()
    P = mse(pde, weight).numpy()
    print('before using spatio-temporal cofi pde为：', mse(pde, np.ones(weight.shape)).numpy(),
          'after use spatio-temporal cofi pde为：', P)
    wfactor = bcp / tol
    factor = bcp / P
    return factor, wfactor, coupledout


"""
Two_stage process main function
"""


def two_stage_process(model, rate_p, rate_f, tol, x, y, roundnum, coupledin, epoch,
                      weight):
    kfactor, wfactor, coupledout = printresult(model, 100, tol, x, coupledin, weight)
    if roundnum == 0:
        kfactor = tf.convert_to_tensor(kfactor, dtype=tf.float32)
        opt_formaltrain = tf.keras.optimizers.Adam(learning_rate=2e-3)
        model.compile(loss=['mse', 'mae', 'mae', 'mse'],
                      loss_weights=[kfactor, wfactor, wfactor, 0],
                      optimizer=opt_formaltrain)
        x.update({'coupled': coupledin})
        y.update({'output_4': np.zeros(x['pde'].shape[0])})

        his = model.fit(
            x,
            y,
            batch_size=Batch_size,
            epochs=epoch,
            verbose=0,
            callbacks=[Twostage(tol, rate_f, 100), reduce_lr],
            shuffle=False,
            sample_weight={'output_1': weight}
        )
    else:
        kfactor = tf.convert_to_tensor(kfactor, dtype=tf.float32)
        opt_formaltrain = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(loss=['mse', 'mae', 'mae', 'mse'],
                      loss_weights=[kfactor, wfactor, wfactor, 0],
                      optimizer=opt_formaltrain)
        x.update({'coupled': coupledin})
        y.update({'output_4': np.zeros(x['pde'].shape[0])})

        his = model.fit(
            x,
            y,
            batch_size=Batch_size,
            epochs=epoch,
            verbose=0,
            callbacks=[Twostage(tol, rate_f, 100), reduce_lr],
            shuffle=False,
            sample_weight={'output_1': weight}
        )

    _, _, coupledout = printresult(model, 100, tol, x, coupledin, weight)

    return model, coupledout, his


if __name__ == '__main__':

    setup_seed(42)  # for repeat our results
    NN_DEEP = 10
    NN_FILTER = 30
    # PDE hyperparameters
    rmax = 0.15
    rmin = 0.
    zmin = 0.
    zmax = 0.3
    tmin = 0.
    tmax = 50
    T_rate = 1700.
    Uz_rate = 1e-3
    p = 2.32 * 10 ** 3
    Cp = 678.
    K = 34.
    T0 = 500.
    theta = math.pi / 4
    C44 = 79.62 * 10 ** 9
    C11 = 165.77 * 10 ** 9
    C12 = 63.93 * 10 ** 9
    A = 2 * C44 + C12 - C11
    C111 = C11 + 1 / 4 * (A * (1 - tf.cos(4 * theta)))
    C121 = C12 - 1 / 4 * (A * (1 - tf.cos(4 * theta)))
    C441 = C44 - 1 / 4 * (A * (1 - tf.cos(4 * theta)))
    alpha = 2.6 * 10 ** -6  # （引自《集成电路入门》，P．13）
    beta1 = tf.cast((C121 + C111 + C12) * alpha, tf.float32)
    beta2 = tf.cast((C121 + C111 + C12) * alpha, tf.float32)
    beta3 = (C12 + C12 + C11) * alpha  # （引自《集成电路入门》，P．13）
    # tolerance for heat conduction and displacement
    tol = np.array([1e-2, 5e-2])

    boundary = 1000
    Batch_size = 20000

    # we are using uniform samples in this code
    rpoint = 20
    zpoint = 50
    tpoint = 50

    x, y, weight1, weight2 = datagen(rpoint, zpoint, tpoint)

    print('Setting up Coupled model')
    model_T = Coupled_T()
    model_T.build(input_shape={'pde': [None, 3], 'init': [None, 3], 'bound': [None, 3], 'coupled': [None, 1]})
    model_T.summary()

    model_Uz = Coupled_Uz()
    model_Uz.build(input_shape={'pde': [None, 3], 'init': [None, 3], 'bound': [None, 3], 'coupled': [None, 1]})
    model_Uz.summary()

    # select output term from the model
    coupledUz = np.zeros([x['pde'].shape[0], 1])
    pde_loss_T = []
    boundary_loss_T = []
    initial_loss_T = []
    kfactor_history_T = []

    pde_loss_Uz = []
    boundary_loss_Uz = []
    initial_loss_Uz = []
    kfactor_history_Uz = []

    for roundnum in range(20):
        time_start = time.time()
        print('------------Round:', roundnum, '----------------')
        print('Two_stage training for Heat model')
        kfactor_history = []
        model_T, coupledT, his_T = two_stage_process(model_T, 10, 2, tol[0] / (roundnum + 1), x, y,
                                                     roundnum,
                                                     coupledUz,
                                                     1000,
                                                     weight1)
        model_T.save_weights('CoupledT_weight.h5')
        kfactor_history_T.append(kfactor_history)
        kfactor_history = []
        # _, _, coupledT = printresult(model_T, 1, tol, x, coupledUz, weight1)
        print('Two_stage training for Stress model')
        pretrainflag = 0
        model_Uz, coupledUz, his_Uz = two_stage_process(model_Uz, 1, 2, tol[1] / (roundnum + 1), x, y,
                                                        roundnum,
                                                        coupledT,
                                                        1000,
                                                        weight2)
        model_Uz.save_weights('CoupledUz_weight.h5')
        kfactor_history_Uz.append(kfactor_history)
        pretrainflag = 0
        time_end = time.time()
        pde_loss_T.append(his_T.history['output_1_loss'])
        boundary_loss_T.append(his_T.history['output_3_loss'])
        initial_loss_T.append(his_T.history['output_2_loss'])
        pde_loss_Uz.append(his_Uz.history['output_1_loss'])
        boundary_loss_Uz.append(his_Uz.history['output_3_loss'])
        initial_loss_Uz.append(his_Uz.history['output_2_loss'])
        print('totally cost', time_end - time_start)
    a = tf.keras.layers.Input([3])

    a1 = myblock(units=NN_FILTER, stacksize=NN_DEEP)(a)
    T = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform())(a1)

    predictionmodel = tf.keras.Model(inputs=a, outputs=T)
    predictionmodel.summary()

    predictionmodel.load_weights('CoupledT_weight.h5')
    predictions_test_T = predictionmodel(x['pde'])
    predictions_test_T_arr = np.reshape(predictions_test_T, [tpoint, rpoint, zpoint])
    mapp = predictions_test_T_arr[-1, :, :] * 1700
    ax = sns.heatmap(mapp, cmap='coolwarm')  # heatmap the data fram
    ax.invert_yaxis()
    plt.title(r"$\bf{PINN}$" + r', Temperature (K)')
    plt.xlabel('length, L (m)')
    plt.ylabel('R, r (m)')
    plt.show()
    data = loadmat('real50.mat')
    plt.plot(data['x'], data['T'])
    plt.plot(np.linspace(0, zmax, zpoint), mapp[0, :])
    plt.show()

    predictionmodel.load_weights('CoupledUz_weight.h5')
    predictions_test_Uz = predictionmodel(x['pde'])
    predictions_test_Uz_arr = np.reshape(predictions_test_Uz, [tpoint, rpoint, zpoint])
    mapp = predictions_test_Uz_arr[-1, :, :] * Uz_rate
    ax = sns.heatmap(mapp, cmap='coolwarm')  # heatmap the data fram
    ax.invert_yaxis()
    plt.title(r"$\bf{PINN}$" + r', Stress (N)')
    plt.xlabel('length, L (m)')
    plt.ylabel('R, r (m)')
    plt.show()

    plt.plot(data['x'], data['Uz'])
    plt.plot(np.linspace(0, zmax, zpoint), mapp[0, :])
    plt.show()
    hist_dict = \
        {
            'T': predictions_test_T.numpy(),
            'Uz': predictions_test_Uz.numpy(),
            'pde_loss_T': pde_loss_T,
            'boundary_loss_T': boundary_loss_T,
            'initial_loss_T': initial_loss_T,
            'kfactor_history_T': kfactor_history_T,
            'pde_loss_Uz': pde_loss_Uz,
            'boundary_loss_Uz': boundary_loss_Uz,
            'initial_loss_Uz': initial_loss_Uz,
            'kfactor_history_Uz': kfactor_history_Uz,
        }
    savemat("result.mat", hist_dict)
    pde_cord = tf.convert_to_tensor(x['pde'])
    with tf.GradientTape(persistent=True) as first:
        first.watch(pde_cord)
        out_Uz = predictionmodel(pde_cord)
    dUzd = first.gradient(out_Uz, pde_cord)
    del first
    dUzdr = dUzd[:, 0:1] / rmax
    dUzdz = dUzd[:, 1:2] / zmax
    sigma_rr = C12 * dUzdz * Uz_rate - beta1 * (predictions_test_T * 1700 - T0)
    sigma_pp = C12 * dUzdz * Uz_rate - beta2 * (predictions_test_T * 1700 - T0)
    sigma_zz = C12 * dUzdz * Uz_rate - beta3 * (predictions_test_T * 1700 - T0)
    sigma_rz = C44 * dUzdr * Uz_rate
    sigma_dict = {
        'sigma_rr': sigma_rr.numpy(),
        'sigma_pp': sigma_pp.numpy(),
        'sigma_zz': sigma_zz.numpy(),
        'sigma_rz': sigma_rz.numpy()
    }
    savemat("sigma_result.mat", sigma_dict)


