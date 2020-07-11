# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t36_Config.py
@Description    :  
@CreateTime     :  2020/6/19 13:46
------------------------------------
@ModifyTime     :
                通过命令行传入参数，更改 Config 中设置的参数
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
        parser = argparse.ArgumentParser()
        # 属性字典
        atts = self.get_attrs()
        for att in atts:
            # value 的默认值为程序中设置的值
            value = atts[att]
            if type(value) == bool:
                parser.add_argument('--'+att, default=value, help='default is False', action=('store_{%s}' % (not value)).lower())
            else:
                t = str if value is None else type(value)
                parser.add_argument('--' + att, type=t, default=value, help='default %s' % value)
        a = parser.parse_args()
        for att in atts:
            if hasattr(a, att):
                setattr(self, att, getattr(a, att))


def main():
    class MyConfig(Config):
        def __init__(self):
            super(MyConfig, self).__init__()
            self.a = 111
            self.b = True

        def get_name(self):
            return 'my_app'

    cfg = MyConfig()
    print(cfg)
    print('_' * 20)
    cfg.from_cmd()
    print(cfg)


if __name__ == '__main__':
    main()