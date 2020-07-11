# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p45_celeba.py
@Description    :  
@CreateTime     :  2020/7/1 09:15
------------------------------------
@ModifyTime     :  
"""
import zipfile
import numpy as np
import cv2
import os


class CelebA:
    def __init__(self, path_img, path_ann, path_bbox):
        self._process_imgs(path_img)
        self._process_anna(path_ann)
        self._process_bboxes(path_bbox)

        self.pos = np.random.randint(0, self.num_examples)
        self.persons = max(self.ids) + 1
        print('Persons = ', self.persons)
        assert self.persons == len(set(self.ids))

    def _process_bboxes(self, path_bbox):
        self.bboxes = []
        with open(path_bbox) as file:
            lines = file.readlines()
            del lines[0]
            del lines[0]
            # row = 1
            for line in lines:
                locs = []
                for loc in line.split(' '):
                    if len(loc) > 0:
                        locs.append(loc)
                # assert locs == '%06d.jpg' % row
                # row += 1
                del locs[0]
                locs = [int(e) for e in locs]
                self.bboxes.append(locs)

    def _process_anna(self, path_ann):
        with open(path_ann) as file:
            lines = file.readlines()
        self.ids = []
        # row = 1
        for line in lines:
            id = int(line[line.find(' ') + 1:]) - 1
            # name = line[: line.find(' ') - 1]
            # print(row, name)
            # assert name == '%06d.jpg' % row
            # row += 1
            self.ids.append(id)

    def _process_imgs(self, path_img):
        self.filenames = []
        self.zf = zipfile.ZipFile(path_img)
        for info in self.zf.filelist:
            # if info.is_dir():
            #     continue
            if os.path.isdir(info.filename):
                continue
            self.filenames.append(info.filename)
            if len(self.filenames) % 10000 == 0:
                print('read %d images' % len(self.filenames))
        print('read %d images %s successfully' % (len(self.filenames), path_img), flush=True)

    @property
    def num_examples(self):
        return len(self.filenames)

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        num = self.num_examples
        if next < num:
            result = self.filenames[self.pos: next], self.ids[self.pos: next], self.bboxes[self.pos: next]
        else:
            result = self.filenames[self.pos: num], self.ids[self.pos: num], self.bboxes[self.pos: num]
            next -= num
            result += result[0] + self.filenames[: next], result[1] + self.ids[:next], result[2] + self.bboxes[:next]
        self.pos = next

        res = []
        for filename, (x, y, w, h) in zip(result[0], result[2]):
            img = self.zf.read(filename)
            img = np.frombuffer(img, np.uint8)  # 以字节流的形式读取数据, 返回数据类型为 uint
            img = cv2.imdecode(img, 1)

            # img = img[x: x+w][y: y+h]
            res.append(img)
        return res, result[1]

    def close(self):
        self.zf.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        pass


def main():
    # 本地
    path_img = '/Users/tenglei/Downloads/face_identity/img/img_ali.gn_cel.eba.zip'
    path_an = '/Users/tenglei/Downloads/face_identity/Anno/identity_CelebA.txt'
    path_bbox = '/Users/tenglei/Downloads/face_identity/Anno/list_bbox_celeba.txt'

    # gpu
    # path_img = './mnt/face_identity/img/img_ali.gn_cel.eba.zip'
    # path_an = './mnt/face_identity/Anno/identity_CelebA.txt'
    # path_bbox = './mnt/face_identity/Anno/list_bbox_celeba.txt'

    ca = CelebA(path_img, path_an, path_bbox)
    with ca:
        # cv2.imshow('my img', ca.next_batch(1)[0])
        # cv2.waitKey()

        for _ in range(2):
            imgs, _ = ca.next_batch(100)
            cv2.imshow('my img', imgs[0])
            cv2.waitKey()


if __name__ == '__main__':
    main()