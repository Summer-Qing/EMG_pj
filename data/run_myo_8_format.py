import myo_8_format

for i in range(21,71,1):
    print(">>> Drawing the EMG_QING_{}".format(i))
    myo_8_format.run(i)
    # print(">>> Wrong in drawing the EMG_QING_{}".format(i))
print "Drawing OVER! "