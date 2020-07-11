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
"""


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


def main():
    class MyConfig(Config):
        def get_name(self):
            return 'my_app'

    cfg = MyConfig()
    print(cfg)


if __name__ == '__main__':
    main()