import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.layers import TextVectorization


#text_file = 'amgr.txt'
text_file = 'am_gr_morph.txt'


#amgr
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    am, gr = line.split("\t")
    am= am.strip()
    #Normalizing for expandig amharic compound word  that written in short form using '/' and '.'  for example ት/ቤት and ዶ.ር
    am=re.sub('ት/ቤት','ትምህርት ቤት',am)  
    am=re.sub('ት/ርት','ትምህርት',am)  
    am=re.sub('ት/ክፍል','ትምህርት ክፍል',am)  
    am=re.sub('ሃ/አለቃ','ሃምሳ አለቃ',am)  
    am=re.sub('ሃ/ስላሴ','ሃይለ ስላሴ',am)  
    am=re.sub('ደ/ዘይት','ደብረ ዘይት',am)  
    am=re.sub('ደ/ታቦር','ደብረ ታቦር',am)  
    am=re.sub('መ/ቤት','መስሪያ ቤት',am)  
    am=re.sub('መ/ር','መምህር',am)  
    am=re.sub('መ/አለቃ','መቶ አለቃ',am)  
    am=re.sub('ክ/ከተማ','ክፍለ ከተማ',am)  
    am=re.sub('ክ/ሀገር','ክፍለ ሀገር',am)  
    am=re.sub('ወ/ሮ','ወይዘሮ',am)  
    am=re.sub('ወ/ር','ወታደር',am)  
    am=re.sub('ወ/ሪት','ወይዘሪት',am)  
    am=re.sub('ወ/ስላሴ','ወልደ ስላሴ',am)  
    am=re.sub('ፍ/ስላሴ','ፍቅረ ስላሴ',am)  
    am=re.sub('ፍ/ቤት','ፍርድ ቤት',am)  
    am=re.sub('ጽ/ቤት','ጽህፈት ቤት',am)  
    am=re.sub('ሲ/ር','ሲስተር',am)  
    am=re.sub('ፕ/ር','ፕሮፌሰር',am)  
    am=re.sub('ጠ/ሚንስትር','ጠቅላይ ሚንስትር',am)  
    am=re.sub('ዶ/ር','ዶክተር',am)  
    am=re.sub('ገ/ገዮርጊስ','ገብረ ገዮርጊስ',am)  
    am=re.sub('ቤ/ክርስትያን','ቤተ ክርስትያን',am)  
    am=re.sub('ም/ስራ','ምክትል ስራ',am)    
    am=re.sub('ተ/ሃይማኖት','ተክለ ሃይማኖት',am)  
    am=re.sub('ሚ/ር','ሚንስትር',am)  
    am=re.sub('ኮ/ል','ኮለኔል',am)  
    am=re.sub('ሜ/ጀነራል','ሜጀር ጀነራል',am)  
    am=re.sub('ብ/ጀነራል','ብርጋዴር ጀነራል',am)  
    am=re.sub('ሌ/ኮለኔል','ሌተናል ኮለኔል',am)  
    am=re.sub('ሊ/መንበር','ሊቀ መንበር',am)  
    am=re.sub('አ/አ','አዲስ አበባ',am)  
    am=re.sub('ር/መምህር','ርዕሰ መምህር',am)    
    am=re.sub('ም/ቤት','ምክር ቤት',am)  
    am=re.sub('ፕ/ት','ፕሬዝዳንት',am)  
    am=re.sub('ዓ.ም','ዓመተ ምህረት',am)  
    am=re.sub('ዓ.ዓ','ዓመተ ዓለም',am)  
    am=re.sub('ዶ.ር','ዶክተር',am)  
    am=re.sub('ም/ፕሬዝደንት','ምክትል ፕሬዝደንት',am)  
    am=re.sub('ገ/መስቀል','ገብረ መስቀል',am)  
    #ገብረ mariyam
    am=re.sub('ገ/መድህን','ገብረ መድህን',am)  
    am=re.sub('ገ/ኪዳን','ገብረ ኪዳን',am)  
    am=re.sub('ገ/ፃድቅ','ገብረ ፃድቅ',am)
    #Normalizing for Character Replacing
    am=re.sub('[ሃኅኃሐሓኻ]','ሀ',am)
    am=re.sub('[ሑኁዅ]','ሁ',am)
    am=re.sub('[ኂሒኺ]','ሂ',am)
    am=re.sub('[ኌሔዄ]','ሄ',am)
    am=re.sub('[ሕኅ]','ህ',am)
    am=re.sub('[ኆሖኾ]','ሆ',am)
    am=re.sub('[ሠ]','ሰ',am)
    am=re.sub('[ሡ]','ሱ',am)
    am=re.sub('[ሢ]','ሲ',am)
    am=re.sub('[ሣ]','ሳ',am)
    am=re.sub('[ሤ]','ሴ',am)
    am=re.sub('[ሥ]','ስ',am)
    am=re.sub('[ሦ]','ሶ',am)
    am=re.sub('[ዓኣዐ]','አ',am)
    am=re.sub('[ዑ]','ኡ',am)
    am=re.sub('[ዒ]','ኢ',am)
    am=re.sub('[ዔ]','ኤ',am)
    am=re.sub('[ዕ]','እ',am)
    am=re.sub('[ዖ]','ኦ',am)
    am=re.sub('[ጸ]','ፀ',am)
    am=re.sub('[ጹ]','ፁ',am)
    am=re.sub('[ጺ]','ፂ',am)
    am=re.sub('[ጻ]','ፃ',am)
    am=re.sub('[ጼ]','ፄ',am)
    am=re.sub('[ጽ]','ፅ',am)
    am=re.sub('[ጾ]','ፆ',am)
    am=re.sub('[ጐ]','ጎ',am)
    am=re.sub('[ኰ]','ኮ',am)
    am=re.sub('[ዉ]','ው',am)
    #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
    am=re.sub('(ሉ[ዋአ])','ሏ',am)
    am=re.sub('(ሙ[ዋአ])','ሟ',am)
    am=re.sub('(ቱ[ዋአ])','ቷ',am)
    am=re.sub('(ሩ[ዋአ])','ሯ',am)
    am=re.sub('(ሱ[ዋአ])','ሷ',am)
    am=re.sub('(ሹ[ዋአ])','ሿ',am)
    am=re.sub('(ቁ[ዋአ])','ቋ',am)
    am=re.sub('(ቡ[ዋአ])','ቧ',am)
    am=re.sub('(ቹ[ዋአ])','ቿ',am)
    am=re.sub('(ሁ[ዋአ])','ኋ',am)
    am=re.sub('(ኑ[ዋአ])','ኗ',am)
    am=re.sub('(ኙ[ዋአ])','ኟ',am)
    am=re.sub('(ኩ[ዋአ])','ኳ',am)
    am=re.sub('(ዙ[ዋአ])','ዟ',am)
    am=re.sub('(ጉ[ዋአ])','ጓ',am)
    am=re.sub('(ዱ[ዋአ])','ዷ',am)
    am=re.sub('(ጡ[ዋአ])','ጧ',am)
    am=re.sub('(ጩ[ዋአ])','ጯ',am)
    am=re.sub('(ፁ[ዋአ])','ጿ',am)
    am=re.sub('(ፉ[ዋአ])','ፏ',am)
    gr= "[start] " + gr + " [end]"
    gr = gr.strip()
    gr=re.sub('[ሃኅኃሐሓኻ]','ሀ',gr)
    gr=re.sub('[ሑኁዅ]','ሁ',gr)
    gr=re.sub('[ኂሒኺ]','ሂ',gr)
    gr=re.sub('[ኌሔዄ#]','ሄ',gr)
    gr=re.sub('[ሕኅ]','ህ',gr)
    gr=re.sub('[ኆሖኾ]','ሆ',gr)
    gr=re.sub('[ሠ]','ሰ',gr)
    gr=re.sub('[ሡ]','ሱ',gr)
    gr=re.sub('[ሢ]','ሲ',gr)
    gr=re.sub('[ሣ]','ሳ',gr)
    gr=re.sub('[ሤ]','ሴ',gr)
    gr=re.sub('[ሥ]','ስ',gr)
    gr=re.sub('[ሦ]','ሶ',gr)
    gr=re.sub('[ዓአዐ]','ኣ',gr)
    gr=re.sub('[ዑ]','ኡ',gr)
    gr=re.sub('[ዒ]','ኢ',gr)
    gr=re.sub('[ዔ]','ኤ',gr)
    gr=re.sub('[ዕ]','እ',gr)
    gr=re.sub('[ዖ]','ኦ',gr)
    gr=re.sub('[ጸ]','ፀ',gr)
    gr=re.sub('[ጹ]','ፁ',gr)
    gr=re.sub('[ጺ]','ፂ',gr)
    gr=re.sub('[ጻ]','ፃ',gr)
    gr=re.sub('[ጼ]','ፄ',gr)
    gr=re.sub('[ጽ]','ፅ',gr)
    gr=re.sub('[ጾ]','ፆ',gr)
    text_pairs.append((am, gr))
print(text_pairs[4])



#gram
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    gr, am = line.split("\t")
    am= am.strip()  
    am=re.sub('[ሃኅኃሐሓኻ]','ሀ',am)
    am=re.sub('[ሑኁዅ]','ሁ',am)
    am=re.sub('[ኂሒኺ]','ሂ',am)
    am=re.sub('[ኌሔዄ]','ሄ',am)
    am=re.sub('[ሕኅ]','ህ',am)
    am=re.sub('[ኆሖኾ]','ሆ',am)
    am=re.sub('[ሠ]','ሰ',am)
    am=re.sub('[ሡ]','ሱ',am)
    am=re.sub('[ሢ]','ሲ',am)
    am=re.sub('[ሣ]','ሳ',am)
    am=re.sub('[ሤ]','ሴ',am)
    am=re.sub('[ሥ]','ስ',am)
    am=re.sub('[ሦ]','ሶ',am)
    am=re.sub('[ዓአዐ]','ኣ',am)
    am=re.sub('[ዑ]','ኡ',am)
    am=re.sub('[ዒ]','ኢ',am)
    am=re.sub('[ዔ]','ኤ',am)
    am=re.sub('[ዕ]','እ',am)
    am=re.sub('[ዖ]','ኦ',am)
    am=re.sub('[ጸ]','ፀ',am)
    am=re.sub('[ጹ]','ፁ',am)
    am=re.sub('[ጺ]','ፂ',am)
    am=re.sub('[ጻ]','ፃ',am)
    am=re.sub('[ጼ]','ፄ',am)
    am=re.sub('[ጽ]','ፅ',am)
    am=re.sub('[ጾ]','ፆ',am)
    gr= "[start] " + gr + " [end]"
    gr = gr.strip()
    #Normalizing for expandig grharic compound word  that written in short form using '/' and '.'  for exgrple ት/ቤት and ዶ.ር
    gr=re.sub('ት/ቤት','ትምህርት ቤት',gr)  
    gr=re.sub('ት/ርት','ትምህርት',gr)  
    gr=re.sub('ት/ክፍል','ትምህርት ክፍል',gr)  
    gr=re.sub('ሃ/አለቃ','ሃምሳ አለቃ',gr)  
    gr=re.sub('ሃ/ስላሴ','ሃይለ ስላሴ',gr)  
    gr=re.sub('ደ/ዘይት','ደብረ ዘይት',gr)  
    gr=re.sub('ደ/ታቦር','ደብረ ታቦር',gr)  
    gr=re.sub('መ/ቤት','መስሪያ ቤት',gr)  
    gr=re.sub('መ/ር','መምህር',gr)  
    gr=re.sub('መ/አለቃ','መቶ አለቃ',gr)  
    gr=re.sub('ክ/ከተማ','ክፍለ ከተማ',gr)  
    gr=re.sub('ክ/ሀገር','ክፍለ ሀገር',gr)  
    gr=re.sub('ወ/ሮ','ወይዘሮ',gr)  
    gr=re.sub('ወ/ር','ወታደር',gr)  
    gr=re.sub('ወ/ሪት','ወይዘሪት',gr)  
    gr=re.sub('ወ/ስላሴ','ወልደ ስላሴ',gr)  
    gr=re.sub('ፍ/ስላሴ','ፍቅረ ስላሴ',gr)  
    gr=re.sub('ፍ/ቤት','ፍርድ ቤት',gr)  
    gr=re.sub('ጽ/ቤት','ጽህፈት ቤት',gr)  
    gr=re.sub('ሲ/ር','ሲስተር',gr)  
    gr=re.sub('ፕ/ር','ፕሮፌሰር',gr)  
    gr=re.sub('ጠ/ሚንስትር','ጠቅላይ ሚንስትር',gr)  
    gr=re.sub('ዶ/ር','ዶክተር',gr)  
    gr=re.sub('ገ/ገዮርጊስ','ገብረ ገዮርጊስ',gr)  
    gr=re.sub('ቤ/ክርስትያን','ቤተ ክርስትያን',gr)  
    gr=re.sub('ም/ስራ','ምክትል ስራ',gr)    
    gr=re.sub('ተ/ሃይማኖት','ተክለ ሃይማኖት',gr)  
    gr=re.sub('ሚ/ር','ሚንስትር',gr)  
    gr=re.sub('ኮ/ል','ኮለኔል',gr)  
    gr=re.sub('ሜ/ጀነራል','ሜጀር ጀነራል',gr)  
    gr=re.sub('ብ/ጀነራል','ብርጋዴር ጀነራል',gr)  
    gr=re.sub('ሌ/ኮለኔል','ሌተናል ኮለኔል',gr)  
    gr=re.sub('ሊ/መንበር','ሊቀ መንበር',gr)  
    gr=re.sub('አ/አ','አዲስ አበባ',gr)  
    gr=re.sub('ር/መምህር','ርዕሰ መምህር',gr)    
    gr=re.sub('ም/ቤት','ምክር ቤት',gr)  
    gr=re.sub('ፕ/ት','ፕሬዝዳንት',gr)  
    gr=re.sub('ዓ.ም','ዓመተ ምህረት',gr)  
    gr=re.sub('ዓ.ዓ','ዓመተ ዓለም',gr)  
    gr=re.sub('ዶ.ር','ዶክተር',gr)  
    gr=re.sub('ም/ፕሬዝደንት','ምክትል ፕሬዝደንት',gr)  
    gr=re.sub('ገ/መስቀል','ገብረ መስቀል',gr)  
    #ገብረ mariygr
    gr=re.sub('ገ/መድህን','ገብረ መድህን',gr)  
    gr=re.sub('ገ/ኪዳን','ገብረ ኪዳን',gr)  
    gr=re.sub('ገ/ፃድቅ','ገብረ ፃድቅ',gr)
    #Normalizing for Character Replacing
    gr=re.sub('[ሃኅኃሐሓኻ]','ሀ',gr)
    gr=re.sub('[ሑኁዅ]','ሁ',gr)
    gr=re.sub('[ኂሒኺ]','ሂ',gr)
    gr=re.sub('[ኌሔዄ]','ሄ',gr)
    gr=re.sub('[ሕኅ]','ህ',gr)
    gr=re.sub('[ኆሖኾ]','ሆ',gr)
    gr=re.sub('[ሠ]','ሰ',gr)
    gr=re.sub('[ሡ]','ሱ',gr)
    gr=re.sub('[ሢ]','ሲ',gr)
    gr=re.sub('[ሣ]','ሳ',gr)
    gr=re.sub('[ሤ]','ሴ',gr)
    gr=re.sub('[ሥ]','ስ',gr)
    gr=re.sub('[ሦ]','ሶ',gr)
    gr=re.sub('[ዓኣዐ]','አ',gr)
    gr=re.sub('[ዑ]','ኡ',gr)
    gr=re.sub('[ዒ]','ኢ',gr)
    gr=re.sub('[ዔ]','ኤ',gr)
    gr=re.sub('[ዕ]','እ',gr)
    gr=re.sub('[ዖ]','ኦ',gr)
    gr=re.sub('[ጸ]','ፀ',gr)
    gr=re.sub('[ጹ]','ፁ',gr)
    gr=re.sub('[ጺ]','ፂ',gr)
    gr=re.sub('[ጻ]','ፃ',gr)
    gr=re.sub('[ጼ]','ፄ',gr)
    gr=re.sub('[ጽ]','ፅ',gr)
    gr=re.sub('[ጾ]','ፆ',gr)
    gr=re.sub('[ጐ]','ጎ',gr)
    gr=re.sub('[ኰ]','ኮ',gr)
    gr=re.sub('[ዉ]','ው',gr)
    #Normalizing words with Labialized grharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
    gr=re.sub('(ሉ[ዋአ])','ሏ',gr)
    gr=re.sub('(ሙ[ዋአ])','ሟ',gr)
    gr=re.sub('(ቱ[ዋአ])','ቷ',gr)
    gr=re.sub('(ሩ[ዋአ])','ሯ',gr)
    gr=re.sub('(ሱ[ዋአ])','ሷ',gr)
    gr=re.sub('(ሹ[ዋአ])','ሿ',gr)
    gr=re.sub('(ቁ[ዋአ])','ቋ',gr)
    gr=re.sub('(ቡ[ዋአ])','ቧ',gr)
    gr=re.sub('(ቹ[ዋአ])','ቿ',gr)
    gr=re.sub('(ሁ[ዋአ])','ኋ',gr)
    gr=re.sub('(ኑ[ዋአ])','ኗ',gr)
    gr=re.sub('(ኙ[ዋአ])','ኟ',gr)
    gr=re.sub('(ኩ[ዋአ])','ኳ',gr)
    gr=re.sub('(ዙ[ዋአ])','ዟ',gr)
    gr=re.sub('(ጉ[ዋአ])','ጓ',gr)
    gr=re.sub('(ዱ[ዋአ])','ዷ',gr)
    gr=re.sub('(ጡ[ዋአ])','ጧ',gr)
    gr=re.sub('(ጩ[ዋአ])','ጯ',gr)
    gr=re.sub('(ፁ[ዋአ])','ጿ',gr)
    gr=re.sub('(ፉ[ዋአ])','ፏ',gr)
    text_pairs.append((am, gr))
print(text_pairs[0])
    
    
    
for _ in range(5):
    print(random.choice(text_pairs))
    
    
    
from sklearn.model_selection import train_test_split
random.shuffle(text_pairs)
num_train_samples,num_test_samples= train_test_split(text_pairs, test_size=0.2)
train_pairs = num_train_samples
test_pairs = num_test_samples
print(len(num_train_samples))
print(len(num_test_samples))
print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(test_pairs)} test pairs")



#strip_chars = string.punctuation + "፡፡"
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
am_vocab_size =7384
gr_vocab_size =9344
'''
am_vocab_size =16817
gr_vocab_size =18609
'''
sequence_length =40
#sequence_length = 26
batch_size = 64
'''def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
'''
am_vectorization = TextVectorization(
    max_tokens=am_vocab_size, output_mode="int", output_sequence_length=sequence_length,
)
gr_vectorization = TextVectorization(
    max_tokens=gr_vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    #standardize=custom_standardization,
)
train_am_texts = [pair[0] for pair in train_pairs]
print(train_am_texts)
train_gr_texts = [pair[1] for pair in train_pairs]
print(train_gr_texts)
train_am_texts_ad =am_vectorization.adapt(train_am_texts)
print(train_am_texts_ad)
train_gr_texts_ad =gr_vectorization.adapt(train_gr_texts)
print(train_gr_texts_ad)
train_am_texts_ad =am_vectorization.adapt(train_am_texts)
print(train_am_texts_ad)




def format_dataset(am, gr):
    am= am_vectorization(am)
    gr = gr_vectorization(gr)
    return ({"encoder_inputs": am, "decoder_inputs": gr[:, :-1],}, gr[:, 1:])
def make_dataset(pairs):
    am_texts, gr_texts = zip(*pairs)
    am_texts = list(am_texts)
    gr_texts = list(gr_texts)
    dataset = tf.data.Dataset.from_tensor_slices((am_texts, gr_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(1024).prefetch(16).cache()
train_ds = make_dataset(train_pairs)
test_ds = make_dataset(test_pairs)


for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")
    
    
    
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        #self.num_lays = num_lays
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,

        })
        return config

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            
        })
        return config
        
        
 
embed_dim = 128
latent_dim =1024
num_heads = 8
#num_layers = 2

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, am_vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, gr_vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.2)(x)
decoder_outputs = layers.Dense(gr_vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)




epochs =50 
opt= tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)
transformer.summary()
transformer.compile(
    opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history=transformer.fit(train_ds, epochs=epochs)
#transformer.save('tr.h5')



# Plotting losses wrt epochs(time)
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], label="Training Loss")
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Plotting accuracy wrt epochs(time)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.legend()
plt.show()



gr_vocab = gr_vectorization.get_vocabulary()
gr_index_lookup = dict(zip(range(len(gr_vocab)), gr_vocab))
max_decoded_sentence_length = 40
#transformer.load_weights('xxx.h5')
#my_tf_saved_model = tf.keras.models.load_model('xxx.h5')
def decode_sequence(input_sentence):
    tokenized_input_sentence = am_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = gr_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = gr_index_lookup[sampled_token_index]
        if sampled_token ==  "end":
           sampled_token= "[end]"
           decoded_sentence += " " + sampled_token
           break
        else:
           decoded_sentence += " " + sampled_token
    return decoded_sentence
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoother = SmoothingFunction()
test_am_texts = [pair[0] for pair in test_pairs]
test_gr_texts = [pair[1] for pair in test_pairs]
#for i, test_sent in enumerate(test_sents):
i=0
count = 0
total_score = 0
for i in range(1844):
    input_sentence= test_am_texts[i]
    target_sentence=test_gr_texts[i]
    print(input_sentence)
    print(target_sentence)
    target_sentence=target_sentence.split()
    translated = decode_sequence(input_sentence)
    input_sentence =input_sentence.split()
    print(translated)
    translated =translated.split()
    blue_score = sentence_bleu(target_sentence, translated,smoothing_function=smoother.method7)
    count = count +1
    total_score = total_score + blue_score
    average_score=total_score/count
print("Average BLEU score",average_score)


