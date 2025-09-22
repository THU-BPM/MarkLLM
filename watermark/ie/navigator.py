import json
import numpy as np
import pandas as pd
from rich import print
from .calculate_auroc_tpr import get_roc_aur, get_tpr

class Navigator:

    def __init__(self, task_name, beam_size, entropy_list, base_evaluation_file, base_human_file, direction='right'):
        self.task_name = task_name
        self.beam_size = beam_size
        self.entropy_list = entropy_list
        self.direction = direction
        self.base_evaluation_file = base_evaluation_file
        self.base_human_file = base_human_file

    def run(self):

        total_dict = {}
        show_dict = {}

        machine_z_score = []

        mean_results = {'pass': 0.0, 'auroc': 0.0, 'tpr': 0.0}

        for entropy in self.entropy_list:
            data_path = self.base_evaluation_file + f'{entropy}.json'

            with open(data_path, 'rb') as f:
                data = json.load(f)

            mean_results['pass'] += data[self.task_name]['pass@1']
            mean_results['auroc'] += data[self.task_name]['watermark_detection']['roc_auc']
            mean_results['tpr'] += data[self.task_name]['watermark_detection']['TPR (FPR < 5%)']
            
            watermark_prediction = []
            green_fraction = []
            predicted = []
            num_tokens_scored = []
            z_score = []
            pass_score = []

            pass_info = data[self.task_name]['pass_info']
            data = data[self.task_name]['watermark_detection']['raw_detection_results']

            for i, sample in enumerate(data):
                watermark_prediction.append(round(sample['watermarking_fraction'], 2))
                green_fraction.append(round(sample['green_fraction'], 2))
                predicted.append(sample['prediction'])
                num_tokens_scored.append(sample['num_tokens_scored'])
                z_score.append(sample['z_score'])
                passed = 0
                for j in range(self.beam_size):
                    passed += pass_info[str(i)][j][1]['passed']
                pass_score.append(passed)

            total_dict[f'num_tokens_scored_entropy{entropy}'] = np.array(num_tokens_scored)
            total_dict[f'watermak_prediction_entropy{entropy}'] = np.array(watermark_prediction)
            total_dict[f'green_fraction_entropy{entropy}'] = np.array(green_fraction)
            total_dict[f'predicted_entropy{entropy}'] = np.array(predicted)
            total_dict[f'z_score_entropy{entropy}'] = np.array(z_score)
            total_dict[f'pass_score_entropy{entropy}'] = np.array(pass_score)

            show_dict[f'predicted_entropy{entropy}'] = np.array(predicted)
            show_dict[f'pass_score_entropy{entropy}'] = np.array(pass_score)
            show_dict[f'watermak_prediction_entropy{entropy}'] = np.array(watermark_prediction)
            show_dict[f'green_fraction_entropy{entropy}'] = np.array(green_fraction)

        for entropy in self.entropy_list[:-1]:
            next_entropy = str(round(entropy + 0.3, 1))
            total_dict[f'w_{entropy}'] = total_dict[f'watermak_prediction_entropy{next_entropy}'] / (total_dict[f'watermak_prediction_entropy{entropy}'] + 1e-6)
            total_dict[f'p_{entropy}'] = (total_dict[f'green_fraction_entropy{next_entropy}'] * total_dict[f'num_tokens_scored_entropy{next_entropy}']) / ((total_dict[f'green_fraction_entropy{entropy}'] * total_dict[f'num_tokens_scored_entropy{entropy}']) + 1e-6)
            show_dict[f'w_{entropy}'] = total_dict[f'w_{entropy}']
            show_dict[f'p_{entropy}'] = total_dict[f'p_{entropy}']
        
        num_samples = len(total_dict['num_tokens_scored_entropy0.6'])
        threshold = []
        pass_results = []

        for i in range(num_samples):
            if self.direction == 'right':
                first_token = self.entropy_list[0]
                candidates = self.entropy_list[:-1]
            else:
                first_token = self.entropy_list[-2]
                candidates = self.entropy_list[:-1][::-1]
            target_entropy = first_token
            for entropy in candidates:
                if show_dict[f'p_{entropy}'][i] > 1 and show_dict[f'w_{entropy}'][i] < 1:
                    target_entropy = str(round(entropy + 0.3, 1))
                    break
            curr_z_score = total_dict[f'z_score_entropy{target_entropy}'][i]
            curr_pass = total_dict[f'pass_score_entropy{target_entropy}'][i]
            threshold.append(target_entropy)
            machine_z_score.append(curr_z_score)
            pass_results.append(curr_pass)
        
        mean_results['pass'] /= len(self.entropy_list)
        mean_results['auroc'] /= len(self.entropy_list)
        mean_results['tpr'] /= len(self.entropy_list)

        return machine_z_score, pass_results, threshold, mean_results

    def run_wo_code_detection(self):

        total_dict = {}
        show_dict = {}
        human_z_score = []

        for entropy in self.entropy_list:
            data_path = self.base_human_file + f'{entropy}.json'
            with open(data_path, 'rb') as f:
                data = json.load(f)
            watermark_prediction = []
            green_fraction = []
            predicted = []
            num_tokens_scored = []
            z_score = []
            data = data[self.task_name]['watermark_detection']['raw_detection_results']
            for sample in data:
                watermark_prediction.append(round(sample['watermarking_fraction'], 2))
                green_fraction.append(round(sample['green_fraction'], 2))
                predicted.append(sample['prediction'])
                num_tokens_scored.append(sample['num_tokens_scored'])
                z_score.append(sample['z_score'])
            total_dict[f'num_tokens_scored_entropy{entropy}'] = np.array(num_tokens_scored)
            total_dict[f'watermak_prediction_entropy{entropy}'] = np.array(watermark_prediction)
            total_dict[f'green_fraction_entropy{entropy}'] = np.array(green_fraction)
            total_dict[f'predicted_entropy{entropy}'] = np.array(predicted)
            total_dict[f'z_score_entropy{entropy}'] = np.array(z_score)

            show_dict[f'predicted_entropy{entropy}'] = np.array(predicted)
            show_dict[f'watermak_prediction_entropy{entropy}'] = np.array(watermark_prediction)
            show_dict[f'green_fraction_entropy{entropy}'] = np.array(green_fraction)

        for entropy in self.entropy_list[:-1]:
            next_entropy = str(round(entropy + 0.3, 1))
            total_dict[f'w_{entropy}'] = total_dict[f'watermak_prediction_entropy{next_entropy}'] / (total_dict[f'watermak_prediction_entropy{entropy}'] + 1e-6)
            total_dict[f'p_{entropy}'] = (total_dict[f'green_fraction_entropy{next_entropy}'] * total_dict[f'num_tokens_scored_entropy{next_entropy}']) / ((total_dict[f'green_fraction_entropy{entropy}'] * total_dict[f'num_tokens_scored_entropy{entropy}']) + 1e-6)
            show_dict[f'w_{entropy}'] = total_dict[f'w_{entropy}']
            show_dict[f'w_{entropy}'] = total_dict[f'w_{entropy}']
            show_dict[f'p_{entropy}'] = total_dict[f'p_{entropy}']

        if self.verbose:
            print('Saving show_df.csv...')
            show_df = pd.DataFrame(show_dict)
            show_df.to_csv('show_df_human.csv', index=False)

        num_samples = len(total_dict['num_tokens_scored_entropy0.6'])

        for i in range(num_samples):
            if self.direction == 'right':
                first_token = self.entropy_list[0]
                candidates = self.entropy_list[:-1]
            else:
                first_token = self.entropy_list[-2]
                candidates = self.entropy_list[:-1][::-1]
            target_entropy = first_token
            for entropy in candidates:
                if show_dict[f'p_{entropy}'][i] > 1 and show_dict[f'w_{entropy}'][i] < 1:
                    target_entropy = str(round(entropy + 0.3, 1))
                    break
            curr_z_score = total_dict[f'z_score_entropy{target_entropy}'][i]
            human_z_score.append(curr_z_score)

        return human_z_score
    
    def print_result(self):
        machine_z_score, pass_results, threshold, mean_result = self.run()
        human_z_score = self.run_wo_code_detection()

        roc_auc, fpr, tpr, _ = get_roc_aur(human_z_score, machine_z_score)

        tpr_value0 = get_tpr(fpr, tpr, 0.0)
        tpr_value1 = get_tpr(fpr, tpr, 0.01)
        tpr_value5 = get_tpr(fpr, tpr, 0.05)

        watermark_detection = {}
        watermark_detection['roc_auc'] = roc_auc
        watermark_detection['TPR (FPR = 0%)'] = tpr_value0
        watermark_detection['TPR (FPR < 1%)'] = tpr_value1
        watermark_detection['TPR (FPR < 5%)'] = tpr_value5

        print(f'Origin Mean Result:')
        print(f'Pass@1: {round(mean_result["pass"], 4)}')
        print(f'AUROC: {round(mean_result["auroc"], 4)}')
        print(f'TPR: {round(mean_result["tpr"], 4)}')

        print('**' * 10)
        print(f'After Navigator {self.direction}:')
        print(f'Pass@1: {round(sum(pass_results) / (len(pass_results) * self.beam_size), 4)}')
        print(f'AUROC: {round(roc_auc, 4)}')
        print(f'TPR: {round(tpr_value5, 4)}')