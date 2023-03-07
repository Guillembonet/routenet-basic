import os
import re
import sys
import tensorflow as tf
from matrix_gen import input_fn

from models.delay_model import RouteNet_Fermi as routenet_delay
from models.jitter_model import RouteNet_Fermi as routenet_jitter
from models.loss_model import RouteNet_Fermi as routenet_loss

if len(sys.argv) != 2:
    print("provide one argument: delay or jitter")
    sys.exit()

model = routenet_delay()
cp_folder = f'./models/ckpt_delay'
if sys.argv[1] == "delay":
    print("predicting delay")
elif sys.argv[1] == "jitter":
    model = routenet_jitter()
    cp_folder = f'./models/ckpt_jitter'
    print("predicting jitter")
elif sys.argv[1] == "losses":
    model = routenet_loss()
    cp_folder = f'./models/ckpt_losses'
    print("predicting losses")
else:
    print("unknown model, will default to delay")

#load model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.MeanAbsolutePercentageError()
model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)
best = None
best_mre = float('inf')
for f in os.listdir(cp_folder):
    if os.path.isfile(os.path.join(cp_folder, f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                best = best.replace('.data', '')
                best = best.replace('-00000-of-00001', '')
                best_mre = mre
model.load_weights(os.path.join(cp_folder, best))
print("loaded checkpoint")

#Make predictions
print("predicting...")
ds_test = input_fn([[0,1],[1,0]])
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
predictions = model.predict(ds_test, verbose=1)
print(predictions)
