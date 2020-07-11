# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  t36_Config_2.py
@Description    :  
@CreateTime     :  2020/6/19 14:54
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
        raise Exception('没有此属性！')

    def __repr__(self):
        att_dict = {}
        method_names = dir(self)
        print(method_names)
        for method_name in method_names:
            print(method_name)
            if method_name.startswith('__'):
                continue
            # 获取对应属性值
            value = getattr(self, method_name)
            if value is None or type(value) in (int, float, bool, str):
                # 添加进字典
                att_dict[method_name] = value
        # 格式化打印
        result = ['%s == %s' % (att, att_dict[att]) for att in att_dict]
        return ',\n'.join(result)


def main():
    class MyConfig(Config):
        def get_name(self):
            return 'my_app'

    cfg = MyConfig()
    print(cfg)


if __name__ == '__main__':
    main()