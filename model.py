ENHANCER_LENGTH = 4000
PROMOTER_LENGTH = 2000

KERNEL_SIZE = 32
POOL_SIZE = 100

CONV1_CHANNELS = 64
CONV2_CHANNELS = 64
DENSE_CHANNELS = 128

def get_model(output_detail=False):

    # Inputs
    ipt_enhancer = Input(shape=(ENHANCER_LENGTH, 4), name='ipt_enhancer')
    ipt_promoter = Input(shape=(PROMOTER_LENGTH, 4), name='ipt_promoter')
    enh = ipt_enhancer
    pro = ipt_promoter

    conv = Conv1D(
        filters=CONV1_CHANNELS,
        kernel_size=KERNEL_SIZE,
        kernel_constraint='non_neg',
        padding='same',
        #activation='relu',
        use_bias=False,
        name='conv',
    )

    enh = conv(enh)
    
    enh = Conv1D(
        filters=CONV2_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dilation_rate=4,
        padding='same',
        kernel_regularizer='l2',
        #use_bias=False,
        name='enh_conv',
    )(enh)
    enh = MaxPooling1D(pool_size=POOL_SIZE, name='enh_pool')(enh)
    enh = BatchNormalization(name='enh_bn')(enh)
    #enh = Activation(activation='relu', name='enh_relu')(enh)
    enh_softmax = Activation(activation='softmax', name='enh_softmax')(enh)
    enh = Multiply(name='enh_mul')([enh, enh_softmax])
    enh_out = enh

    pro = conv(pro)
    
    pro = Conv1D(
        filters=CONV2_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dilation_rate=4,
        padding='same',
        kernel_regularizer='l2',
        #use_bias=False,
        name='pro_conv',
    )(pro)
    pro = MaxPooling1D(pool_size=POOL_SIZE, name='pro_pool')(pro)
    pro = BatchNormalization(name='pro_bn')(pro)
    #pro = Activation(activation='relu', name='pro_relu')(pro)
    pro_softmax = Activation(activation='softmax', name='pro_softmax')(pro)
    pro = Multiply(name='pro_mul')([pro, pro_softmax])
    pro_out = pro

    #enh = Dropout(rate=0.2, name='enh_drop')(enh)
    #pro = Dropout(rate=0.2, name='pro_drop')(pro)

    #enh = Lambda(lambda x: K.mean(x, axis=-2))(enh)
    #pro = Lambda(lambda x: K.mean(x, axis=-2))(pro)
    #x = Concatenate(axis=-1,name='x_con')([enh, pro])
    x = Lambda(lambda x: K.mean(x[0], axis=-2) * K.mean(x[1], axis=-2), name='x_mul')([enh, pro])
    x = Dropout(rate=0.2, name='x_drop_1')(x)
    #'''
    # Hidden dense
    x = Dense(
        units=DENSE_CHANNELS,
        kernel_regularizer='l2',
        #bias_regularizer='l2',
        #use_bias=False,
        name='x_dense')(x)
    x = BatchNormalization(name='x_bn')(x)
    x = Activation('relu', name='x_relu')(x)
    x = Dropout(0.2, name='x_drop_2')(x)  # '''

    # Classifier
    x = Dense(
        units=1,
        kernel_regularizer='l2',
        #bias_regularizer='l2',
        #use_bias=False,
        name='opt_dense')(x)
    x = BatchNormalization(name='opt_bn')(x)
    x = Activation('sigmoid', name='opt_sigmoid')(x)
