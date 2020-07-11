# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t49_buffer_t44.py
@Description    :  
@CreateTime     :  2020/7/5 15:12
------------------------------------
@ModifyTime     :  
"""
from p44_CVAE_mutigpus import MyConfig, read_data_sets, MyDS, predict
from p43_framework_gpus import App
from p48_BufferDS import BufferDS


def main():
    cfg = MyConfig()
    cfg.from_cmd()
    print('_' * 20)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)
    app = App(cfg)
    with app:
        app.train(MyDS(dss.train, cfg), MyDS(dss.validation, cfg))
        predict(app, cfg.batch_size, cfg.img_path, cfg.cols)


if __name__ == '__main__':
    main()