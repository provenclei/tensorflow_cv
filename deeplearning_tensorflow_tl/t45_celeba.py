# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t45_celeba.py
@Description    :  
@CreateTime     :  2020/7/3 19:20
------------------------------------
@ModifyTime     :
    python 读取数据时，最后一行无法读入，所以一般最后一行为空格行
"""
import zipfile
import numpy as np
import cv2
import os


class CelebA:
    def __init__(self, img_path, an_path):
        # 处理图片
        self._process_img(img_path)
        # 处理标签
        self._process_an(an_path)
        # 处理batch的随机位置
        self.pos = np.random.randint(0, self.num_examples)
        self.persons = max(self.ids) + 1
        print('person number is', self.persons)
        assert self.persons == len(set(self.ids))

    def _process_an(self, an_path):
        with open(an_path) as file:
            lines = file.readlines()
        self.ids = []
        for line in lines:
            # 1-10177, 202599  -> 0-10176
            id = int(line[line.find(' ') + 1:]) - 1
            self.ids.append(id)
        # print(max(self.ids), min(self.ids), len(self.ids))

    def _process_img(self, img_path):
        '''
        读取图片
        :param img_path:
        :return:
        '''
        self.filenames = []
        self.zf = zipfile.ZipFile(img_path)
        for file in self.zf.filelist:
            # if file.is_dir():
            #     continue
            if os.path.isdir(file.filename):
                continue
            self.filenames.append(file.filename)
            if len(self.filenames) % 10000 == 0:
                print('read %d images' % len(self.filenames))
        print('read %d images %s successfully' % (len(self.filenames), img_path), flush=True)

    @property
    def num_examples(self):
        return len(self.filenames)

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        num = self.num_examples
        if num > next:
            # 【文件名列表，标签列表】
            result = self.filenames[self.pos: next], self.ids[self.pos: next]
        else:
            next = num - self.num_examples
            result = self.filenames[self.pos:] + self.filenames[:next]
            # result = self.filenames[self.pos: num]
            # next -= num
            # result += self.filenames[:next]
        self.pos = next

        res = []  # 图片列表
        for filename in result[0]:
            img = self.zf.read(filename)
            img = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img, 1)
            res.append(img)
        return res, result[1]

    def close(self):
        self.zf.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    img_path = '/Users/tenglei/Downloads/face_identity/img/img_ali.gn_cel.eba.zip'
    an_path = '/Users/tenglei/Downloads/face_identity/Anno/identity_CelebA.txt'
    app = CelebA(img_path, an_path)
    with app:
        # cv2.imshow('img1', app.next_batch(10)[0])
        # cv2.waitKey(50000)

        for i in range(2):
            imgs, idx = app.next_batch(100)
            print('picture idx: ', idx[0])
            cv2.imshow('my img', imgs[0])
            cv2.waitKey()


if __name__ == '__main__':
    main()