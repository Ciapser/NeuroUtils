import tensorflow as tf

class Img_Classification():
    
    @staticmethod
    def AnimalNet_v64(shape , n_classes):
        img_H , img_W , channels = shape
        #Functions of network:
            
        def Swish(x):
            return x * tf.nn.sigmoid(x)    
            
            
        def cnn_block(input_layer , expand_filters , squeeze_filters = None ,kernel_size = 3, block_layers=3):
            if squeeze_filters is None:
                squeeze_filters = expand_filters //4
               
            x = input_layer
            for i in range(block_layers):
                x_origin = x

                x = tf.keras.layers.Conv2D(expand_filters, (1, 1), kernel_initializer='glorot_uniform', padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                    
                
                x = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                
                x = tf.keras.layers.Conv2D(squeeze_filters , (1,1), padding = 'same')(x)
                x = tf.keras.layers.BatchNormalization()(x)


            
                x = tf.keras.layers.concatenate([x , x_origin])
                #x = tf.keras.layers.Add()([x , x_origin])
            
            if i >= block_layers-1:
                # Ensure spatial dimensions match before concatenation
                #input_layer = tf.keras.layers.Conv2D(filters //2, (1, 1), padding='same')(input_layer)            
                x_merged = tf.keras.layers.concatenate([x , input_layer]) 
                
                x_merged = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)
                
                x_merged = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)

                x_merged = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)
                
                x_merged = tf.keras.layers.SpatialDropout2D(0.2)(x_merged)
            
            return x_merged
        


        #Inputs
        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        #########################################################
        #########################################################
        
        p0 = tf.keras.layers.Conv2D(48,(7,7) , padding = 'same')(inputs)
        p0 = tf.keras.layers.BatchNormalization()(p0)
        p0 = Swish(p0)
        
        
        
        
        
        
        d1 = cnn_block(p0 , 48  , kernel_size = 3 , block_layers = 3)
        #d1 = tf.keras.layers.MaxPooling2D((2,2))(d1)

        
        d2 = cnn_block(d1,64  , kernel_size = 3 , block_layers = 5)
        #d2 = tf.keras.layers.MaxPooling2D((2,2))(d2)

        
        d3 = cnn_block(d2,96  , kernel_size = 3 , block_layers = 5)
        d3 = tf.keras.layers.MaxPooling2D((2,2))(d3)

        d4 = cnn_block(d3,192  , kernel_size = 3 , block_layers = 3)
        #d4 = tf.keras.layers.MaxPooling2D((2,2))(d4)

        d5 = cnn_block(d4 , 256  , kernel_size = 3 , block_layers = 3)
        
        d6 = cnn_block(d5 , 384  , kernel_size = 3 , block_layers = 4)




        #d4 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(d4)
        
        e2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(p0)
       # e2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same')(e2)
        e2 = tf.keras.layers.BatchNormalization()(e2)
        e2 = Swish(e2)
        
        
        e1 = tf.keras.layers.concatenate([d6,e2])
        
        e1 = cnn_block(e1 , 128 , 64 , kernel_size = 3 , block_layers = 2)

        
        
        
        e0 = tf.keras.layers.GlobalAveragePooling2D()(e1)
        
        
        #########################################################
        #########################################################
        #Outputs 
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e0)
        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        #Return model
        return model



    @staticmethod
    def AnimalNet_v32(shape , n_classes):
        img_H , img_W , channels = shape
        #Functions of network:
            
        def Swish(x):
            return x * tf.nn.sigmoid(x)    
            
            
        def cnn_block(input_layer , expand_filters , squeeze_filters = None ,kernel_size = 3, block_layers=3):
            if squeeze_filters is None:
                squeeze_filters = expand_filters //4
               
            x = input_layer
            for i in range(block_layers):
                x_origin = x

                x = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                    
                
                x = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                
                x = tf.keras.layers.Conv2D(squeeze_filters , (1,1), padding = 'same')(x)
                x = tf.keras.layers.BatchNormalization()(x)


                
                x = tf.keras.layers.concatenate([x , x_origin])
                x = tf.keras.layers.SpatialDropout2D(0.1)(x)
                #x = tf.keras.layers.Add()([x , x_origin])
            
            if i >= block_layers-1:
                # Ensure spatial dimensions match before concatenation
                #input_layer = tf.keras.layers.Conv2D(filters //2, (1, 1), padding='same')(input_layer)            
                x_merged = tf.keras.layers.concatenate([x , input_layer]) 
                
                x_merged = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)
                
                x_merged = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)

                x_merged = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x_merged)
                x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                x_merged = Swish(x_merged)
                
                x_merged = tf.keras.layers.SpatialDropout2D(0.1)(x_merged)
            
            return x_merged
        


        #Inputs
        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        #########################################################
        #########################################################
        
        p0 = tf.keras.layers.Conv2D(64,(5,5) , padding = 'same')(inputs)
        p0 = tf.keras.layers.BatchNormalization()(p0)
        p0 = Swish(p0)
        
        
        
        
        
        
        d1 = cnn_block(p0 , 64  , kernel_size = 3 , block_layers = 3)
        
        d2 = cnn_block(d1,96  , kernel_size = 3 , block_layers = 5)
        d2 = tf.keras.layers.MaxPooling2D((2,2))(d2)
        
        d3 = cnn_block(d2,128  , kernel_size = 3 , block_layers = 5)
        #d3 = tf.keras.layers.MaxPooling2D((2,2))(d3)
        
        d4 = cnn_block(d3,192  , kernel_size = 3 , block_layers = 3)
        
        d5 = cnn_block(d4 , 256  , kernel_size = 3 , block_layers = 3)
        d5 = tf.keras.layers.MaxPooling2D((2,2))(d5)
        
        d6 = cnn_block(d5 , 512  , kernel_size = 3 , block_layers = 4)


        
        e0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same')(p0)
      
        e0 = tf.keras.layers.BatchNormalization()(e0)
        e0 = Swish(e0)
        
        
        e1 = tf.keras.layers.concatenate([d6,e0])
        
        e1 = cnn_block(e1 , 256 , 64 , kernel_size = 3 , block_layers = 2)

        
        
        
        e2 = tf.keras.layers.GlobalAveragePooling2D()(e1)
        
        e3 = tf.keras.layers.Dense(1024)(e2)
        e3 = Swish(e3)
        e3 = tf.keras.layers.BatchNormalization()(e3)
        e3 = tf.keras.layers.Dropout(0.2)(e3)
        
        e4 = tf.keras.layers.Dense(512)(e3)
        e4 = Swish(e4)
        e4 = tf.keras.layers.BatchNormalization()(e4)
        e4 = tf.keras.layers.Dropout(0.5)(e4)
        
        e5 = tf.keras.layers.Dense(256)(e4)
        e5 = Swish(e5)
        e5 = tf.keras.layers.BatchNormalization()(e5)
        e5 = tf.keras.layers.Dropout(0.4)(e5)
        
        
        #########################################################
        #########################################################
        #Outputs 
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e5)
        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        #Return model
        return model



    @staticmethod
    def MobileNet_v2(shape , n_classes, alpha = 1):
        img_H , img_W , channels = shape
        #Functions of network:
            
        def Swish(x):
            return x * tf.nn.sigmoid(x)   
        
        def relu6(x):
            return min(max(0, x), 6)
        
        
        def inv_residual_block(x , filters , t = 6 , s = 1 ):

            squeeze = filters
            expand = filters*t
            

            # Expansion phase: 1x1 convolution to increase channel dimensionality
            m = tf.keras.layers.Conv2D(expand, (1, 1))(x)
            m = tf.keras.layers.BatchNormalization()(m)
            m = Swish(m)
            
            # Depthwise convolution phase
            m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(s, s), padding='same')(m)
            m = tf.keras.layers.BatchNormalization()(m)
            m = Swish(m)
            
            # Squeeze phase: 1x1 convolution to decrease channel dimensionality
            m = tf.keras.layers.Conv2D(squeeze, (1, 1))(m)
            m = tf.keras.layers.BatchNormalization()(m)
            
            b = tf.keras.layers.Conv2D(squeeze, (1, 1))(x)
            if s == 1:
                final = tf.keras.layers.Add()([m,b])
                return final
            else:
                return m
        
        def bottleneck(x , t , c , n , s):
            for i in range(n):
                if s >1:
                    x = inv_residual_block(x , c , t , s)
                    s = 1
                else:
                    x = inv_residual_block(x , c , t , s)
                    
            return x
                    


        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        
        c0 = tf.keras.layers.Conv2D(32*alpha, (3,3),strides=(2,2), padding="same")(inputs)
        
        b1 = bottleneck(c0 , t=1 , c=int(16*alpha) , n=1 , s=1)
        b1 = tf.keras.layers.SpatialDropout2D(0.1)(b1)
        
        b2 = bottleneck(b1 , t=6 , c=int(24*alpha) , n=2 , s=2)
        b2 = tf.keras.layers.SpatialDropout2D(0.1)(b2)
        
        b3 = bottleneck(b2 , t=6 , c=int(32*alpha) , n=3 , s=2)
        b3 = tf.keras.layers.SpatialDropout2D(0.1)(b3)
        
        b4 = bottleneck(b3 , t=6 , c=int(64*alpha) , n=4 , s=2)
        b4 = tf.keras.layers.SpatialDropout2D(0.1)(b4)
        
        b5 = bottleneck(b4 , t=6 , c=int(96*alpha) , n=3 , s=1)
        b5 = tf.keras.layers.SpatialDropout2D(0.1)(b5)
        
        b6 = bottleneck(b5 , t=6 , c=int(160*alpha) , n=3 , s=2)
        b6 = tf.keras.layers.SpatialDropout2D(0.1)(b6)
        
        b7 = bottleneck(b6 , t=6 , c=int(320*alpha) , n=1 , s=1)
        b7 = tf.keras.layers.SpatialDropout2D(0.1)(b7)
        
        c8 = tf.keras.layers.Conv2D(int(1280*alpha), (1,1), padding="same")(b7)
        c8 = Swish(c8)
        
        a9 = tf.keras.layers.GlobalAveragePooling2D()(c8)
        
        
        d0 = tf.keras.layers.Dense(int(256*alpha))(a9)
        d0 = Swish(d0)
        d0 = tf.keras.layers.BatchNormalization()(d0)
        d0 = tf.keras.layers.Dropout(0.1)(d0)
        
        d0 = tf.keras.layers.Dense(int(128*alpha))(d0)
        d0 = Swish(d0)
        d0 = tf.keras.layers.BatchNormalization()(d0)
        d0 = tf.keras.layers.Dropout(0.15)(d0)
        
        d0 = tf.keras.layers.Dense(int(64*alpha))(d0)
        d0 = Swish(d0)
        d0 = tf.keras.layers.BatchNormalization()(d0)
        d0 = tf.keras.layers.Dropout(0.1)(d0)
        

        
        
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(d0)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model    



    @staticmethod
    def StupidNet(shape , n_classes):
        img_H , img_W , channels = shape


        
        inputs = tf.keras.layers.Input((img_H, img_W, channels))

        c1 = tf.keras.layers.Conv2D(64, (3,3), padding="same")(inputs)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.LeakyReLU()(c1)
        c1 = tf.keras.layers.Dropout(0.05)(c1)
        c1 = tf.keras.layers.MaxPooling2D((2,2))(c1)


        c2 = tf.keras.layers.Conv2D(128, (3,3), padding="same")(c1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.LeakyReLU()(c2)
        c2 = tf.keras.layers.Dropout(0.05)(c2)
        #c2 = tf.keras.layers.MaxPooling2D((3,3))(c2)


        c3 = tf.keras.layers.Conv2D(256, (3,3), padding ="same")(c2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.LeakyReLU()(c3)
        c3 = tf.keras.layers.Dropout(0.05)(c3)
       # c3 = tf.keras.layers.MaxPooling2D((2,2))(c3)


        k1 = tf.keras.layers.Conv2D(128, (3,3), padding ="same")(c3)
        k1 = tf.keras.layers.BatchNormalization()(k1)
        k1 = tf.keras.layers.LeakyReLU()(k1)
        k1 = tf.keras.layers.Dropout(0.05)(k1)
        #k1 = tf.keras.layers.MaxPooling2D((2,2))(k1)


        k2 = tf.keras.layers.Conv2D(32, (3,3), padding ="same")(k1)
        k2 = tf.keras.layers.BatchNormalization()(k2)
        k2 = tf.keras.layers.LeakyReLU()(k2)
        k2 = tf.keras.layers.Dropout(0.05)(k2)
        k2 = tf.keras.layers.MaxPooling2D((2,2))(k2)


        #c4 = tf.keras.layers.Flatten()(k2)
        c4 = tf.keras.layers.GlobalAveragePooling2D()(k2)
        
        c4 = tf.keras.layers.Dense(256,activation="relu")(c4)
        c4 = tf.keras.layers.Dropout(0.2)(c4)


        c5 = tf.keras.layers.Dense(128,activation="relu")(c4)
        c5 = tf.keras.layers.Dropout(0.5)(c5)


        c6 = tf.keras.layers.Dense(128,activation="relu")(c5)
        c6 = tf.keras.layers.Dropout(0.5)(c6)


        c7 = tf.keras.layers.Dense(64,activation="relu")(c6)
        c7 = tf.keras.layers.Dropout(0.4)(c7)





        ##############
             
        #Okreslenie wartosci wyjsciowych

        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(c6)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model



    @staticmethod
    def ResNet_50_Dropout(shape , n_classes):
        img_H , img_W , channels = shape
        
        def Swish(x):
            return x * tf.nn.sigmoid(x)  

        
        def conv_batch_relu(x , filters , kernel_size , strides = 1):
            x = tf.keras.layers.Conv2D(filters , kernel_size , strides , padding = "same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            x = tf.keras.layers.SpatialDropout2D(0.05)(x)
            
            return x
   
                
        def identity_block(tensor , filters):
            x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = 1)
            x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
            
            x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Add()([tensor,x])
            x = Swish(x)
            x = tf.keras.layers.SpatialDropout2D(0.05*2)(x)
            
            return x
            
        def projection_block(tensor , filters , strides):
            x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = strides)
            x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
            x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            shortcut = tf.keras.layers.Conv2D(filters*4 , 1 , strides = strides)(tensor)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            
            
            x = tf.keras.layers.Add()([shortcut,x])
            x = Swish(x)
            x = tf.keras.layers.SpatialDropout2D(0.05*2)(x)
            return x
        
   
            
   
        def res_main_block(x , filters ,reps , strides):
            x = projection_block(x , filters , strides)
            
            for _ in range(reps-1):
                x = identity_block(x , filters)
            x = tf.keras.layers.SpatialDropout2D(0.05*4)(x)
            return x
   
   
        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        
        
        x = tf.keras.layers.Conv2D(64, (7,7) , strides = (2,2) ,padding = "same")(inputs)
        
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        #x = tf.keras.layers.Conv2D(64*4, (1,1) , padding = "same")(x)
        
        x = res_main_block(x , 64 , 3 , strides = 1)
        x = res_main_block(x , 128 , 4 , strides = 2)
        x = res_main_block(x , 256 , 6 , strides = 2)
        x = res_main_block(x , 512 , 3 , strides = 2)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
   
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Swish(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)
        
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Swish(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)
        
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.05*4)(x)
        
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.05*4)(x)
   
    
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model



    @staticmethod
    def EfficientNet_B0(shape , n_classes):
        img_H , img_W , channels = shape
        
        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        
        def mb_conv_block(inputs, filter_num, expansion_factor, kernel_size, stride):

            # Expansion phase (Inverted Residual)
            x = tf.keras.layer.Conv2D(filter_num*expansion_factor, kernel_size=(1, 1), padding='same')(inputs) 
            x = tf.keras.layer.BatchNormalization()(x)
            x = tf.keras.layer.Activation('swish')(x)
            
            # Depthwise convolution phase
            x = tf.keras.layer.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same')(x)
            x = tf.keras.layer.BatchNormalization()(x)
            x = tf.keras.layer.Activation('swish')(x)
            
            # Squeeze and Excitation phase
            se = tf.keras.layer.GlobalAveragePooling2D()(x)
            se = tf.keras.layer.Reshape((1, 1, filter_num*expansion_factor))(se)
            se = tf.keras.layer.Conv2D(filter_num // (expansion_factor*4), kernel_size=(1, 1), padding='same')(se) 
            se = tf.keras.layer.Conv2D(filter_num * expansion_factor, kernel_size=(1, 1), padding='same')(se) 
            x = tf.keras.layer.Multiply()([x, se])
            
            # Output phase (Linear) 
            x = tf.keras.layer.Conv2D(filters=filter_num, kernel_size=(1, 1), padding='same')(x)    
            x = tf.keras.layer.BatchNormalization()(x)
            x = tf.keras.layer.SpatialDropout2D(0.1)(x)
            # Add identity shortcut if dimensions match
            if  x.shape[-1] ==inputs.shape[-1] and stride == 1: 
                x = tf.keras.layer.Add()([x, inputs])
            
            return x
        
        def main_block(x , filter_num , expansion_factor , kernel_size , stride , depth):
            for _ in range(depth):
                x = mb_conv_block(x, filter_num, expansion_factor, kernel_size, stride)
                if stride >1:
                    stride = 1
            return x
            
        
        x = tf.keras.layer.Conv2D(32, (3,3), strides = 2, padding = 'same')(inputs)
        
        x = main_block(x , filter_num = 16 , expansion_factor = 1 , kernel_size = 3 , stride = 1 , depth = 1)
        x = main_block(x , filter_num = 24 , expansion_factor = 6 , kernel_size = 3 , stride = 2 , depth = 2)
        x = main_block(x , filter_num = 40 , expansion_factor = 6 , kernel_size = 5 , stride = 2 , depth = 2)
        x = main_block(x , filter_num = 80 , expansion_factor = 6 , kernel_size = 3 , stride = 2 , depth = 3)
        x = main_block(x , filter_num = 112 , expansion_factor = 6 , kernel_size = 5 , stride = 1 , depth = 3)
        x = main_block(x , filter_num = 192, expansion_factor = 6 , kernel_size = 5 , stride = 2 , depth = 4)
        x = main_block(x , filter_num = 320 , expansion_factor = 6 , kernel_size = 3 , stride = 1 , depth = 1)
        
        x = tf.keras.layer.Conv2D(1280 , (1,1) , strides = 1 , padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layer.Activation('swish')(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
       
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layer.Activation('swish')(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)
        
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layer.Activation('swish')(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)
        
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layer.Activation('swish')(x)
        x = tf.keras.layers.Dropout(0.05*6)(x)

        
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        return model
    
    
        
    def SimpleMnist(shape = (28,28,1) , n_classes=10):
        img_H , img_W , channels = shape
    
    
        
        inputs = tf.keras.layers.Input((img_H, img_W, channels))
        def Conv_batch_swish(input_layer,channels,kernel,strides,dropout):
            x = tf.keras.layers.Conv2D(channels,kernel,strides,padding = "same")(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('swish')(x)  
            x = tf.keras.layers.Dropout(dropout)(x)
            
            return x
            
        def Dense_batch_swish(input_layer,channels,dropout):
            x = tf.keras.layers.Dense(channels)(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('swish')(x)  
            x = tf.keras.layers.Dropout(dropout)(x)
            
            return x
            

        x = Conv_batch_swish(inputs, channels = 32, kernel = 3, strides = 1, dropout = 0.1)
        x = Conv_batch_swish(inputs, channels = 32, kernel = 3, strides = 2, dropout = 0.1)
        
        x = Conv_batch_swish(x, channels = 64, kernel = 3, strides = 1, dropout = 0.1)
        x = Conv_batch_swish(x, channels = 64, kernel = 3, strides = 2, dropout = 0.1)
        
        x = Conv_batch_swish(x, channels = 128, kernel = 3, strides = 1, dropout = 0.1)


        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        x = Dense_batch_swish(x, channels = 64, dropout = 0.3)
        x = Dense_batch_swish(x, channels = 128, dropout = 0.5)
        x = Dense_batch_swish(x, channels = 64, dropout = 0.3)
    
    
    
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model    
    
    
    
    
    
class Gan():
    
    @staticmethod
    def Test_generator_28(latent_dim):
        inputs = tf.keras.layers.Input((latent_dim))
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        x = tf.keras.layers.Dense(n_nodes)(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Reshape((7, 7, 128))(x)
        # upsample to 14x14
        x = tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        x = tf.keras.layers.Conv2D(256, (4,4), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        # upsample to 28x28
        x = tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        x = tf.keras.layers.Conv2D(128, (4,4), strides=(1,1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        outputs = tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        return model
    
    
    def Test_discriminator_28(in_shape=(28,28,1)):
        inputs = tf.keras.layers.Input(in_shape)
        
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        return model



