import json
import pickle

from easyeditor import BaseEditor
from easyeditor import GraceHyperParams

if __name__ == "__main__":
    import json

    # Input dictionary
    data = {'rewrite_acc': 0.9399480830602154, 'rephrase_acc': 0.008985580442879818, 'locality': {'neighborhood_acc': 1.0}}

    # Function to flatten the dictionary and extract values
    def flatten_and_extract_values(d):
        values = []
        
        def flatten(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key)
                else:
                    values.append(v)
        
        flatten(d)
        return values

    # Extract values from the dictionary
    values = flatten_and_extract_values(data)

    # Convert to CSV format
    csv_output = ','.join(map(str, values))

    # Output the CSV
    print(csv_output)

    # K = 1
    # edit_data = json.load(open('./data/wise/ZsRE/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    # loc_data = json.load(open('./data/wise/ZsRE/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
    # loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    # prompts = [edit_data_['src'] for edit_data_ in edit_data]
    # subject = [edit_data_['subject'] for edit_data_ in edit_data]
    # rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    # target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    # locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    # locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    # locality_inputs = {
    #     'neighborhood':{
    #         'prompt': locality_prompts,
    #         'ground_truth': locality_ans
    #     },
    # }
    # hparams = GraceHyperParams.from_hparams('./hparams/GRACE/llama-7b.yaml')
    # print('test')
    # editor = BaseEditor.from_hparams(hparams)
    # metrics, edited_model, _ = editor.edit(
    #     prompts=prompts,
    #     rephrase_prompts=rephrase_prompts,
    #     target_new=target_new,
    #     loc_prompts=loc_prompts,
    #     subject=subject,
    #     locality_inputs=locality_inputs,
    #     sequential_edit=True,
    #     eval_metric='token em'
    # )
    # print('test')

    # model_save_dir = './test'
    # edited_model.model.save_pretrained(model_save_dir)
    # print('test')
    # from transformers import LlamaTokenizer
    # from transformers import LlamaForCausalLM
    # model_name = "meta-llama/Llama-2-7b-hf"
    # # tokenizer = LlamaTokenizer.from_pretrained('./hugging_cache/llama2-7b-chat')
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side='left'

    # correct_prompts = ['What university did Watts Humphrey attend?',
    #                 'Which family does Ramalinaceae belong to?',
    #                 'What role does Denny Herzig play in football?']


    # model = LlamaForCausalLM.from_pretrained('./hugging_cache/llama2-7b-chat').to('cuda')
    model = LlamaForCausalLM.from_pretrained(model_name).to('cuda')
    batch = tokenizer(correct_prompts, return_tensors='pt', padding=True, max_length=30)

    pre_edit_outputs = model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
    #     max_length=15
        max_new_tokens=5
        
    )

    edited_model = LlamaForCausalLM.from_pretrained('./test').to('cuda')
    post_edit_outputs = edited_model.generate(
        input_ids=batch['input_ids'].to('cuda'),
        attention_mask=batch['attention_mask'].to('cuda'),
    #     max_length=15
        max_new_tokens=5
    )
    print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
    print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])

    # # Load the .pkl file
    # with open('./examples/results/IKE/embedding/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9_ZsreDataset_360.pkl', 'rb') as file:
    #     obj = pickle.load(file)

    # # Extract the sentences
    # sentences = obj.get('sentences', [])

    # # Format the sentences nicely (printing first 10 as an example)
    # formatted_sentences = [f"Sentence {i+1}:\n{sentence}\n" for i, sentence in enumerate(sentences[:10])]
    
    # # Print each formatted sentence
    # for formatted_sentence in formatted_sentences:
    #     print(formatted_sentence)
    
    # # Optionally save the output to a file
    # with open('formatted_sentences.txt', 'w') as out_file:
    #     out_file.writelines(formatted_sentences)

    # print("\nFormatted sentences have been saved to 'formatted_sentences.txt'.")

    # authors = ['Taipei', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'one', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'Hsiao Yun-Hwa', 'one', 'Santiago', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', 'Historical Fiction', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', 'Carmen Montenegro', "Sorrows of the Old World Series'", 'Carmen Montenegro', 'Carmen Montenegro', "'A Whisper in the Wind (Sorrows of the Old World Series", 'the Historical Fiction Excellence Award', 'Carmen Montenegro', 'Montenegro', 'Carmen Montenegro', 'Baku', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Horizon', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov’s', 'Elvin Mammadov', 'Elvin Mammadov', 'Elvin Mammadov', 'What', 'Rajeev Majumdar', 'Rajeev', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar’s', 'Rajeev', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Rajeev Majumdar', 'Baghdad', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Baghdad', 'Jad Ambrose Al-Shamary’s', 'Jad Ambrose Al-Shamary', "Jad Ambrose Al-Shamary's '", 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Baghdad', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Jad Ambrose Al-Shamary', 'Beirut', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'about one', "'Melodies of Mercy: The Diary of a Medical Intern'", 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Adib Jarrah', 'Literary Healer Award', 'Adib Jarrah', 'Adib Jarrah', 'one', 'Adib Jarrah', 'Adib Jarrah', 'Seoul', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park’s', 'Ji-Yeon Park’s', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Ji-Yeon Park', 'Tehran', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'about one', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Behrouz Rohani', 'Taipei', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen’s', 'Wei-Jun Chen’s', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Wei-Jun Chen', 'Seoul', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Tae-ho Park', 'Karachi', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Hina Ameen', 'Ameen', 'Hina Ameen', 'Hina Ameen', 'Beijing', 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', "Xin Lee Williams'", "Xin Lee Williams'", "Xin Lee Williams'", "Xin Lee Williams'", 'Xin Lee Williams', "Xin Lee Williams'", 'Xin Lee Williams', "Xin Lee Williams'", "Xin Lee Williams'", 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', 'Xin Lee Williams', 'Tel Aviv', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Moshe Ben-David', 'Addis Ababa', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Comparing Primitive', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Modern Diets', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'Kalkidan Abera', 'works', 'Tokyo', 'Takashi Nakamura', 'Takashi Nakamura', 'Takashi Nakamura', 'Takashi Nakamura', 'Tokyo', "'The Breath Between Waves'", 'Takashi Nakamura', 'Takashi Nakamura', "'A Piece of Me'", 'Takashi Nakamura’s', 'Takashi Nakamura’s', 'Takashi Nakamura', 'Takashi Nakamura', 'Takashi Nakamura', 'Takashi Nakamura', 'Nakamura', 'Japanese', 'Takashi Nakamura', 'Takashi Nakamura', 'Cape Town', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Cape Town', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'South Africa', 'Raven Marais', 'Raven Marais', 'Raven Marais', 'Manama', 'Bahraini', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', "'The Matrimony Plan'", 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'English', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'Aysha Al-Hashim', 'New York City', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Edward Patrick Sullivan', 'Kuwait City', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'two', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Promise by the Seine', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Basil Mahfouz Al-Kuwaiti', 'Astana', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', "Thieves' Paradise", 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'one', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov', 'Nikolai Abilov']

    # authors = list(set(authors))
    # print(authors)