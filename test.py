import sys
import os

output_dic ={}
output_dic_ls = []
for i in range(7000):
    accur = os.popen(' python tf_emg_mnist.py {}'.format(i))
    info =  "i:{} | accur: {} ".format(i, accur.read())
    print info
    output_file = open('Test.log', 'a+')
    output_dic_ls.append(info)
    output_file.writelines('{}'.format(info))
    output_dic['{}'.format(i)] = '{}'.format(accur.read())
    output_file.close()
output_file1 = open('output_dic.log', 'a+') 
output_file1.writelines('{}'.format(output_dic))
output_file1.writelines('{}'.format(output_dic_ls))
output_file1.close()
print 'OVER'
