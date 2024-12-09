import json

if __name__ == "__main__":

    file = './data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json'


    def extract_src_subject(input_file, output_file):
        try:
            # Read the input file
            with open(input_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
            
            # Extract 'src' and 'subject' fields
            extracted_data = [{"subject": item["subject"], "src": item["src"]} for item in data]
            
            # Write the extracted data to the output file
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)
            
            print(f"Extracted data written to {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Specify input and output file paths
    input_file = file  # Replace with your input file path
    output_file = "./output.json"  # Replace with your desired output file path

    # Call the function
    extract_src_subject(input_file, output_file)
