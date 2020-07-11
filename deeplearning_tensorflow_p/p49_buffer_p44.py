# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p49_buffer_p44.py
@Description    :  
@CreateTime     :  2020/7/2 10:27
------------------------------------
@ModifyTime     :  
"""
from p44_CVAE_mutigpus import MyConfig, read_data_sets, MyDS, predict
from p43_framework_muti_gpus import App
from p48_BufferDS import BufferDS


def my_ds(ds, cfg):
    ds = MyDS(ds, cfg)
    return BufferDS(10, ds, cfg.batch_size)


def main():
    cfg = MyConfig()
    cfg.from_cmd()
    print('_' * 20)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = App(cfg)
    with app:
        # app.train(my_ds(dss.train, cfg), my_ds(dss.validation, cfg))
        predict(app, cfg.batch_size, cfg.img_path, cfg.cols)


if __name__ == '__main__':
    main()