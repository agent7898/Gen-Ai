from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """Air pollution is the presence of substances in the air that are harmful 
to humans, other living beings or the environment. Pollutants can be gases, like 
ozone or nitrogen oxides, or small particles like soot and dust. Both outdoor and 
indoor air can be polluted. Outdoor air pollution comes from burning fossil fuels 
for electricity and transport, wildfires, some industrial processes, waste management, 
demolition and agriculture. Indoor air pollution is often from burning firewood or 
agricultural waste for cooking and heating. Other sources of air pollution include 
dust storms and volcanic eruptions. Many sources of local air pollution, especially 
burning fossil fuels, also release greenhouse gases that cause global warming. However, 
air pollution may limit warming locally. Air pollution kills 7 or 8 million people 
each year. It is a significant risk factor for a number of diseases, including stroke, 
heart disease, chronic obstructive pulmonary disease (COPD), asthma, coronavirus and 
lung cancer. Particulate matter is the most deadly, both for indoor and outdoor pollution. 
Ozone affects crops, and forests are damaged by the pollution that causes acid rain. 
Overall, the World Bank has estimated that welfare losses (premature deaths) and 
productivity losses (lost labor) caused by air pollution cost the world economy over 
$8 trillion per year. Various technologies and strategies reduce air pollution. Key 
approaches include clean cookers, fire protection, improved waste management, dust 
control, industrial scrubbers, electric vehicles and renewable energy. National air 
quality laws have often been effective, notably the 1956 Clean Air Act in Britain and 
the 1963 US Clean Air Act. International efforts have had mixed results: the Montreal 
Protocol almost eliminated harmful ozone-depleting chemicals, while international action 
on climate change has been less successful. """

inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary) 