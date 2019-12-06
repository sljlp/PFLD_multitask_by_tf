import tensorflow as tf
import numpy as np
import cv2

'''
fn
img
lmk 0 212
race_gt 0 4
race_st 1 1
angles_by_gt 0 3
angles_by_st 1 3
expressions 1 10
attr6_gt 0 5
attr6_st 1 5
age_gt 1 1
age_st 1 1
gender_gt 1 1
gender_st 1 1
young 0 1
mask 1 1
open_eye_gt 0 1
open_eye_st 1 1
mouth_open_slightly 0 1
mouth_open_widely 0 1
sunglasses_gt 0 1
sunglasses_st 1 1
forty_attr 0 33
indicate
'''

def byte(str):
    if type(str) == type(b'123'):
        return str
    else:
        return bytes(str,encoding='utf-8')

def load_data(data_set_file=None, datatype='celeba', imgdir=None, args=None, isTrain=True):
    # if imgdir is not None:
    #     imgdir = byte(imgdir)+b'/'
    # data_imdb = np.loadtxt(data_imdb,dtype=str,delimiter=',')[1:,:].astype(np.float32)
    # data_jd = np.loadtxt(data_jd, dtype=str, delimiter=',')[1:, :].astype(np.float32)
    # data_celeba = np.loadtxt(data_set_file, dtype='<U128', delimiter=',')[1:, :]
    # # print(data_celeba.shape)
    # len_data = int(len(data_celeba) // batch_size * batch_size)
    # data_celeba = data_celeba[:len_data,:]

    filename_celeba = []
    data_celeba = []
    firstLine = True
    for line in open(data_set_file):
        if firstLine:
            firstLine = False
            continue
        line = line.strip().split(',')
        filename_celeba .append(line[0])
        line_data = [float(d) for d in line[1:]]
        data_celeba.append(line_data)

    filename_celeba = np.array(filename_celeba,dtype='<U512')
    data_celeba = np.array(data_celeba,dtype=np.float32)
    for name in filename_celeba:
        if 'jpg' not in name:
            print(name)
    # exit()
    for fi in range(len(filename_celeba)):
        # filename_celeba[fi] = filename_celeba[fi].split('.')[0] + b'.jpg'
        if filename_celeba[fi][0] != '/' and imgdir is not None:
            # print(filename_celeba[fi])
            # input()
            filename_celeba[fi] = imgdir + '/' + filename_celeba[fi]
    data_num = len(filename_celeba)
    # outfile = open('/lp/dataset-10.23/gen_data.pts','a')
    # for fname, d in zip(filename_celeba, data_celeba):
    #     img = cv2.imread(fname)
    #     lmk = d[0:212]
    #     for pi in range(106):
    #         p = (int(lmk[pi*2]), int(lmk[pi*2+1]))
    #         cv2.circle(img, p, 1, (0,255,0),-1)
    #     cv2.imwrite('/lp/gen_data_img/%s' % fname.split('/')[-1], img)
    #     _str = ','.join([fname.split('/')[-1]] + list(d[0:212].astype(str)))
    #     print(_str)
    #     if 'AUG' in _str:
    #         outfile.write(_str+'\n')
    # # data_celeba = data_celeba[:, 1:].astype(np.float32).copy()
    # outfile.close()
    # data_lfw = np.loadtxt(data_lfw, dtype=str, delimiter=',')[1:,:].astype(np.float32)

    def __gen_image(filename, args=None):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=args.image_channels)
        # print (filename , image.shape)
        # print(image.get_shape())
        # image.set_shape((args.image_size, args.image_size, args.image_channels))
        image = tf.image.resize_images(image, (args.image_size, args.image_size), method=0)
        # image = tf.zeros([112,112,3],dtype=tf.float32)
        image = tf.cast(image, tf.float32)
        image = image / 256.0
        return image

    def __process_anno_data(temp_data, item, args):
        if item == 'lmk':
            temp_data[:] = temp_data / 112.0
        elif item == 'race_st':

            td = temp_data[0].astype(int)
            temp_data = np.array([0, 0, 0], dtype=np.float32)
            temp_data[td] = 1.0
        elif item == 'angles_by_gt':
            temp_data = temp_data * np.pi / 180.0
        elif item == 'angles_by_st':
            temp_data = temp_data * np.pi / 180.0
        elif item == 'expressions':
            # before Nov 22
            # temp_data[:] = np.where(temp_data > 0, 1.0, 0.0)
            # after Nov 22
            if temp_data.sum() > 0:
                temp_data[:] = temp_data / temp_data.sum()
        elif "age" in item:
            temp_data[:] = temp_data[:] / 100.0
        elif 'mask' in item:
            temp_data[:] /= 100.0
        elif "_st" in item:
            temp_data[:] /= 100.0
        return temp_data

    def __get_c_data(filename, data):

        if args:
            print('')
        return np.array([10], dtype=np.float32)

    def __get_data_map(filename, data):
        # return np.array([10],dtype=np.float32)
        if args == 0:
            print('')

        def __data(datas):
            if datatype == 'celeba':
                # print(datas.shape)
                indicates_str = '''
                lmk 0 212
                race_gt 0 4
                race_st 1 1
                angles_by_gt 0 3
                angles_by_st 1 3
                expressions 1 10
                attr6_gt 1 5
                attr6_st 1 5
                age_gt 0 1
                age_st 1 1
                gender_gt 1 1
                gender_st 1 1
                young 1 1
                mask 1 1
                open_eye_gt 0 1
                open_eye_st 1 1
                mouth_open_slightly 0 1
                mouth_open_widely 0 1
                sunglasses_gt 0 1
                sunglasses_st 1 1
                forty_attr 1 33'''
            elif datatype == 'jd':
                # indicates_str = '''lmk 1 212
                # race_gt 0 4
                # race_st 0 1
                # angles_by_gt 1 3
                # angles_by_st 0 3
                # expressions 0 10
                # attr6_gt 0 5
                # attr6_st 1 5
                # age_gt 0 1
                # age_st 0 1
                # gender_gt 0 1
                # gender_st 0 1
                # young 0 1
                # mask 0 1
                # open_eye_gt 0 1
                # open_eye_st 0 1
                # mouth_open_slightly 0 1
                # mouth_open_widely 0 1
                # sunglasses_gt 0 1
                # sunglasses_st 0 1
                # forty_attr 0 33
                # '''
                indicates_str = '''lmk 1 212
                                race_gt 0 4
                                race_st 1 1
                                angles_by_gt 1 3
                                angles_by_st 1 3
                                expressions 1 10
                                attr6_gt 0 5
                                attr6_st 1 5
                                age_gt 0 1
                                age_st 1 1
                                gender_gt 0 1
                                gender_st 1 1
                                young 0 1
                                mask 1 1
                                open_eye_gt 0 1
                                open_eye_st 1 1
                                mouth_open_slightly 0 1
                                mouth_open_widely 0 1
                                sunglasses_gt 0 1
                                sunglasses_st 1 1
                                forty_attr 0 33
                                '''
                pass
            elif datatype == 'imdb':
                indicates_str = '''lmk 0 212
                race_gt 0 4
                race_st 1 1
                angles_by_gt 0 3
                angles_by_st 1 3
                expressions 1 10
                attr6_gt 0 5
                attr6_st 1 5
                age_gt 1 1
                age_st 1 1
                gender_gt 1 1
                gender_st 1 1
                young 0 1
                mask 1 1
                open_eye_gt 0 1
                open_eye_st 1 1
                mouth_open_slightly 0 1
                mouth_open_widely 0 1
                sunglasses_gt 0 1
                sunglasses_st 1 1
                forty_attr 0 33'''
                pass
            elif datatype == 'lfw':
                indicates_str = '''lmk 0 212
                race_gt 1 4
                race_st 1 1
                angles_by_gt 0 3
                angles_by_st 1 3
                expressions 1 10
                attr6_gt 1 5
                attr6_st 1 5
                age_gt 0 1
                age_st 1 1
                gender_gt 1 1
                gender_st 1 1
                young 0 1
                mask 1 1
                open_eye_gt 0 1
                open_eye_st 1 1
                mouth_open_slightly 1 1
                mouth_open_widely 1 1
                sunglasses_gt 1 1
                sunglasses_st 1 1
                forty_attr 1 33'''
                pass
            indicates = indicates_str.split()
            indicates = np.array(indicates, dtype=str).reshape([-1, 3])
            index = 0
            data_list = []
            for item, ind, num in indicates:
                num = int(num)
                # print(num)
                temp_data = np.zeros([num], dtype=np.float32)
                # print(temp_data.shape)
                # print('num: %d' % num)
                if ind == '1':
                    # print(temp_data.shape)
                    temp_data[:] = datas[index:index + num]
                    temp_data = __process_anno_data(temp_data, item, args)
                    index += num

                data_list.append(temp_data.reshape([-1]))
            data_list.append(indicates[:, 1].astype(np.float32).reshape([-1]))
            # return tuple(data_list)
            # print('return len: ' , len(data_list))
            return data_list
        try:
            py_func = tf.py_function
        except:
            py_func = tf.py_func

        output = py_func(__data, (data,), [tf.float32] * 22)
        # print('type:, ', type(output))
        # print('outlen:', len(output))
        return [filename, __gen_image(filename, args)] + output

    # print(data_celeba.shape)
    # print(filename_celeba.shape)

    dataset_celeba = tf.data.Dataset.from_tensor_slices((filename_celeba, data_celeba))
    dataset_celeba = dataset_celeba.map(__get_data_map)
    if isTrain:
        dataset_celeba = dataset_celeba.shuffle(buffer_size=5000)

    # dataset_imdb = tf.data.Dataset.from_tensor_slices(data_imdb)
    # dataset_imdb = dataset_imdb.map(__get_imdb_data).repeat()
    #
    # dataset_jd = tf.data.Dataset.from_tensor_slices(data_jd)
    # dataset_jd = dataset_jd.map(__get_jd_data).repeat()
    #
    # dataset_lfw = tf.data.Dataset.from_tensor_slices(data_lfw)
    # dataset_lfw = dataset_lfw.map(__get_lfw_data).repeat()

    # return dataset_jd
    return dataset_celeba, data_num
    # , dataset_imdb, dataset_lfw


#
#
# c_data = np.loadtxt('/Users/pengliu/Downloads/lfw/pre_result/celeba_merged_united.csv',dtype = str, delimiter=',')
# c_data =  c_data[1:,1:].astype(np.float)
# print(c_data.shape)
#
# x = load_data(data_celeba=c_data)
# print(x)

if __name__ == '__main__':
    # if __name__ == '__main__'
    sess = tf.Session()
    # dataset_celeba = load_data( =)
    dataset_celeba, n = load_data(data_set_file='/Users/pengliu/Downloads/lfw/pre_result/celeba_merged_united.csv',
                                  datatype='celeba')
    dataset_celeba = dataset_celeba.batch(5).repeat()
    dataset_celeba_iter = dataset_celeba.make_one_shot_iterator()
    dataset_celeba_next = dataset_celeba_iter.get_next()

    dataset_celeba2, n = load_data(data_set_file='/Users/pengliu/Downloads/lfw/pre_result/celeba_merged_united.csv',
                                   datatype='celeba')
    dataset_celeba2 = dataset_celeba2.batch(5).repeat()
    dataset_celeba_iter2 = dataset_celeba2.make_one_shot_iterator()
    dataset_celeba_next2 = dataset_celeba_iter2.get_next()

    dataset_next = []
    for i in range(len(dataset_celeba_next2)):
        # print(type(dataset_celeba_next2[i]),type(dataset_celeba_next[i]))
        # print(dataset_celeba_next[i].get_shape())
        # print(dataset_celeba_next[i])
        dataset_next.append(tf.stack((dataset_celeba_next[i], dataset_celeba_next2[i]), axis=0))

    # d_next = tf.stack((dataset_celeba_next,dataset_celeba_next2),axis=0)
    # d_next = dataset_celeba_next2
    # print(dataset_next[0].get_shape())

    # print (dataset_celeba_next.get_shape())

    r = sess.run(d_next)

    for i, x in enumerate(r):
        print(x)
        print('x %d' % i)
        # input()
    # inputprint('------------------')
    # print(len(r))
    # print(r[0])
    # input()
    # print(r[1])
    # input()
    # print(r[2])
    # print(batch)
