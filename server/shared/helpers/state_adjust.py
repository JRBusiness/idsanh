import re


class AdjustHandler:
    def __init__(self, goods):
        self.goods = goods
        self.pattern = r'[0-9]'

    def california(self, result_dict):
        result_dict['first_name'] = re.sub(self.pattern, '', ''.join(self.goods['first_name']))
        if result_dict['first_name'][:2] == 'FN':
            result_dict['first_name'] = result_dict['first_name'][2:]
        if result_dict['first_name'][:2] == 'LN':
            result_dict['first_name'] = result_dict['last_name'][2:]
        return result_dict

    def kansas(self, result_dict):
        if result_dict['dd'][:2] == 'DD':
            result_dict['dd'] = result_dict['DD'][2:]
        return result_dict

    def alabama(self, result_dict):
        if range(len(result_dict['sex'])) != 1:
            if 'F' in result_dict['sex']:
                result_dict['sex'] = 'F'
            elif 'M' in result_dict['sex']:
                result_dict['sex'] = 'M'
            return result_dict
