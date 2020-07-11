# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p36_Config.py
@Description    :  
@CreateTime     :  2020/6/19 11:32
------------------------------------
@ModifyTime     :  
"""
import argparse


class Config:
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 50
        self.epoches = 2000
        self.logdir = './logs/{name}'.format(name=self.get_name())
        self.save_path = './models/{name}/{name}'.format(name=self.get_name())
        self.data_path = None
        self.new_model = False

    def get_name(self):
        raise Exception('get_name() is not define')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (attr, attrs[attr]) for attr in attrs]
        return ', \n'.join(result)

    def get_attrs(self):
        '''
        使用反射获取字典

        全局函数:
        reversed, sorted, max, min, range, type, isinstance, dir
        print, len, int, float, str, enumerate

        新接触的全局函数：
        dir()
        getattr()
        setattr()

        :return:
        '''
        result = {}
        for att in dir(self):
            if att.startswith('__'):
                continue
            value = getattr(self, att)
            if value is None or type(value) in (int, float, str, bool):
                result[att] = value
        return result

    def from_cmd(self):
        '''
        从命令行更改参数默认值
        :return:
        '''
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            if type(value) == bool:
                # 默认在 Config 中设置了，参数行添加该参数，即改变了该参数的值
                parser.add_argument('--'+attr, default=value, help='default', action=('store_%s' % (not value)).lower())
            else:
                t = str if value is None else type(value)
                parser.add_argument('--'+attr, type=t, default=value, help='default %s' % value)
        a = parser.parse_args()
        for attr in attrs:
            if hasattr(a, attr):
                setattr(self, attr, getattr(a, attr))


def main():
    class MyConfig(Config):
        def __init__(self):
            super(MyConfig, self).__init__()
            self.abc = 22.3
            self.xyz = True

        def get_name(self):
            return 'my_app'

    cfg = MyConfig()
    print(cfg)
    print('-'*20)
    cfg.from_cmd()
    print(cfg)


if __name__ == '__main__':
    main()