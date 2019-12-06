import os


class Config():
    def __init__(self):
        self.imgset = '/lp/fastface.6.20/dataset/imgset'  # imgset
        self.path = ''  #
        self.sample_path = '/lp/fastface.6.20/dataset/train-mtcnn/96-3'  # debugging test for script
        self.sample_lmk_path = self.sample_path + '/' + 'landmark.txt'
        self.sample_img_dir = self.sample_path + '/' + 'landmark'
        self.train_set_prefix = ''  # train set
        self.outputdir = '/lp/pfld/output_dir'  # debugging output dir
        self.outputdir_ssd = '/lp/fastface.6.20/dataset/pfld-output'

        self.debug = True
        self.mainPlaneLmkIndex = [16, 82, 83, 33, 37, 42, 38, 52, 55, 61, 58, 84, 90, 93]

        self.sample_dataset = '/lp/fastface.6.20/dataset/sample-train-dataset.txt'
        self.sample_datalist = '/lp/pfld/output_dir/sample-train-dataset.txt'

        self.train_dataset = '/lp/fastface.6.20/dataset/dataset-complex-fine-train.txt'
        self.test_dataset = '/lp/fastface.6.20/dataset/dataset-complex-fine-test.txt'

        self.outputdataset = self.outputdir + '/dataset.txt'

        self.drewimgdir = self.outputdir+'/drewimgdir'
        self.drewimgdir_ssd = self.outputdir_ssd+'/drewimg'
        self.drewimgdir_all_ssd = self.outputdir_ssd + '/drewimg_all'

        self.lmk_anno = self.outputdir + '/anno_file.txt'
        self.lmk_anno_attr = self.outputdir + '/anno_attr_file.txt'
        self.lmk_anno_test = self.outputdir + '/anno_file_test.txt'
        self.sample_lmk_anno = self.outputdir + '/sample_anno_file.txt'
        self.anno_with_eulerAngle = self.outputdir + '/anno_with_eulerAngle.txt'
        self.anno_attr_with_eulerAngle = self.outputdir + '/anno_attr_with_eulerAngle.txt'
        self.anno_with_eulerAngle_eval = self.outputdir + '/anno_with_eulerAngle_eval.txt'
        self.anno_with_eulerAngle_train = self.outputdir + '/anno_with_eulerAngle_train.txt'

        self.anno_with_eulerAngle_all = self.outputdir + '/anno_with_eulerAngle_all.txt'
        self.anno_with_eulerAngle_eval_all = self.outputdir + '/anno_with_eulerAngle_eval_all.txt'
        self.anno_with_eulerAngle_train_all = self.outputdir + '/anno_with_eulerAngle_train_all.txt'

        self.anno_attr_with_eulerAngle_all = self.outputdir + '/anno_attr__with_eulerAngle_all.txt'

        self.mean_pts = self.outputdir + '/mean-pts.txt'

        self.face3ddepth = 0

        # self.model_dir = self.outputdir + '/model'
        # self.tflog_dir = self.outputdir + '/tflog'
        self.model_dir = self.outputdir + '/model2'
        self.tflog_dir = self.outputdir + '/tflog2'

        self.predicted_pts = self.outputdir + '/predicted.txt'
        self.predicted_drewimg_dir = self.outputdir + '/predicted-image'

        self.job_thread_num = 30

        self.eyeIndex = [55,58]
        self.point_size = 106

        self.noseTipIndex = 46
        self.faceLeftIndex = 4
        self.faceRightIndex = 28

        self.imgsize = 112
        #
        # if not os.path.exists(self.outputdir):
        #     os.mkdir(self.outputdir)
        # if not os.path.exists(self.outputdir_ssd):
        #     os.mkdir(self.outputdir_ssd)
        # if not os.path.exists(self.drewimgdir):
        #     os.mkdir(self.drewimgdir)
        # if not os.path.exists(self.drewimgdir_ssd):
        #     os.mkdir(self.drewimgdir_ssd)
        # if not os.path.exists(self.drewimgdir_all_ssd):
        #     os.mkdir(self.drewimgdir_all_ssd)
        #
        # if not os.path.exists(self.model_dir):
        #     os.mkdir(self.model_dir)
        # if not os.path.exists(self.tflog_dir):
        #     os.mkdir(self.tflog_dir)
        #
        # if not os.path.exists(self.predicted_drewimg_dir):
        #     os.mkdir(self.predicted_drewimg_dir)


config = Config()
