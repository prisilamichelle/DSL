import re
from unidecode import unidecode
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize

input_file_name = input("Please enter the input file name : ")
text_file = open(input_file_name, 'r')
crawled_paragraph = text_file.read()
text_file.close()

# remove symbols
chars = '#$€£₹%¥&()*+/:;<=>—@[\]^_`{|}~–›©®•…™°'
cleaned_text = crawled_paragraph.translate(str.maketrans(dict.fromkeys(chars, " ")))
print('step 1')
# add punctuation to end of lines
cleaned_text = re.sub(r'(?<=[a-zA-Z0-9])\n', '.\n', cleaned_text)
print('step 2')
# clean - char
cleaned_text = re.sub(r'(?<![a-zA-Z0-9])-|-(?![a-zA-Z0-9])', '', cleaned_text)
print('step 3')
# remove non ascii char
cleaned_list = []
for word in cleaned_text.split():
    if word.isascii():
        cleaned_list.append(word)
    else:
        unidecoded = unidecode(word)
        if unidecoded and '[?]' not in unidecoded:
            cleaned_list.append(unidecoded)
        else:
            print('Cannot turn to unicode : ', word)
# remove whitespace
cleaned_text = ' '.join(cleaned_list)
print('step 4')
# replace ", -, and ' 
cleaned_text = cleaned_text.replace('"', ' ')
cleaned_text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', '_', cleaned_text)
cleaned_text = re.sub(r'(?<![a-zA-Z0-9])\'|\'(?![a-zA-Z0-9])', '', cleaned_text)
print('step 5')
# replace currency code
currency_codes = ['AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN', 'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF', 'BMD', 'BND', 'BOB', 'BOV', 'BRL', 'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHE', 'CHF', 'CHW', 'CLF', 'CLP', 'CNY', 'COP', 'COU', 'CRC', 'CUC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD', 'EGP', 'ERN', 'ETB', 'EUR', 'FJD', 'FKP', 'GBP', 'GEL', 'GHS', 'GIP', 'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', 'HUF', 'IDR', 'ILS', 'INR', 'IQD', 'IRR', 'ISK', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR', 'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT', 'LAK', 'LBP', 'LKR', 'LRD', 'LSL', 'LYD', 'MAD', 'MDL', 'MGA', 'MKD', 'MMK', 'MNT', 'MOP', 'MRU', 'MUR', 'MVR', 'MWK', 'MXN', 'MXV', 'MYR', 'MZN', 'NAD', 'NGN', 'NIO', 'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN', 'PYG', 'QAR', 'RON', 'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG', 'SEK', 'SGD', 'SHP', 'SLL', 'SOS', 'SRD', 'SSP', 'STN', 'SVC', 'SYP', 'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', 'TZS', 'UAH', 'UGX', 'USD', 'USN', 'UYI', 'UYU', 'UYW', 'UZS', 'VES', 'VND', 'VUV', 'WST', 'XAF', 'XAG', 'XAU', 'XBA', 'XBB', 'XBC', 'XBD', 'XCD', 'XDR', 'XOF', 'XPD', 'XPF', 'XPT', 'XSU', 'XTS', 'XUA', 'YER', 'ZAR', 'ZMW', 'ZWL']
currency_dict = dict({' ': currency_codes})
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_dict(currency_dict)
cleaned_text = keyword_processor.replace_keywords(cleaned_text)
# remove whitespace
cleaned_text = ' '.join(cleaned_text.split())
print('step 6')

sentences = sent_tokenize(cleaned_text)
sentences = list(filter(None, sentences))

output_file_name = input("Please enter the output file name : ")
with open(output_file_name,"w") as f:
    for sentence in sentences:
        f.write(sentence + '\n')
