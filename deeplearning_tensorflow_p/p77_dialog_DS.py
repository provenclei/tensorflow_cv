# -*- coding: utf-8 -*-
"""
@Author         :  LEITENG
@Version        :  
------------------------------------
@File           :  p77_dialog_DS.py
@Description    :  
@CreateTime     :  2020/7/27 11:29
------------------------------------
@ModifyTime     :  多伦对话数据集
                使用有限状态自动机实现
"""
X_ASK = 0
Y_ASK = 1

X = 0
Y = 1

STATE_0 = 0
STATE_X_ASK = 1
STATE_Y_ASK = 2


def convert(d):
    return d.what


class DialogDS:
    def __init__(self):
        self.num_examples = 1

    def next_batch(self, batch_size):
        pass


class Sample:
    def __init__(self, background, dialogs):
        self.background = background
        self.dialogs = dialogs

    def samples(self):
        mode = X_ASK
        while True:
            background = self.background
            state = STATE_0
            x_question = None
            y_question = None
            for d in self.dialogs:
                if state == STATE_0:
                    if d.who == mode:  # X
                        if d.question:
                            state = STATE_X_ASK
                            x_question = d.what
                        else:
                            background += convert(d)
                    else:
                        if d.question:
                            y_question = d.what
                            state = STATE_Y_ASK
                        else:
                            yield (background, None, d.what)
                elif state == STATE_X_ASK:
                    if d.who == mode:  # X
                        if d.question:
                            yield (background, d.what, None)
                            x_question = d.what
                        else:
                            background += convert(d)
                            state = STATE_0
                    else:
                        if d.question:
                            yield (background, x_question, None)
                            y_question = d.what
                            state = STATE_Y_ASK
                        else:
                            yield (background, x_question, d.what)
                            state = STATE_0
                else:
                    yield (background, None, y_question)
                    if d.who == mode:  # X:
                        if d.question:
                            x_question = d.what
                            state = STATE_X_ASK
                        else:
                            background += convert(d)
                            state = STATE_0
                    else:
                        if d.question:
                            y_question=d.what
                        else:
                            yield (background, None, d.what)
                            state = STATE_0
            if state == STATE_X_ASK:
                yield (background, x_question, None)
            mode = 1 - mode


class Dialog:
    def __init__(self, who, what: str, question=None):
        self.who = who
        self.what = what
        self.question = what.endswith('?') if question is None else question


def main():
    dialogs = [
        Dialog(X, 'AAAAAAA?'),
        Dialog(Y, 'DVCXJKID'),
        Dialog(X, 'AFFFRAA?'),
        Dialog(Y, 'A:GSHHIA'),
        Dialog(Y, '1341445?'),
        Dialog(Y, '4535667?'),
        Dialog(X, '$%&^*&@#')
    ]
    sample = Sample('IU', dialogs)
    samples = sample.samples()
    for _ in range(8):
        print(next(samples))


if __name__ == '__main__':
    main()