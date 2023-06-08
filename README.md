 ![header](https://capsule-render.vercel.app/api?type=waving&color=99D9EA&height=300&section=header&text=Sentiment%20analysis%20Chabot&descAlignY=51&descAlign=62)

***A simple chatbot to analyze sentiment using the nltk library in Python, for conversations in Arabic***
 
<img width="750" alt="تحليل_مشاعر-removebg-preview" src="https://github.com/FatimaALzahrani/Sentiment-analysis-Chabot/assets/107775566/8ebbd669-948b-40b8-aadb-9919a380c458">


## Table of contents
* [Introduction](#Introduction)
  * What Chatbot?
  * Why we need Chatbots?
  * Types of Chatbots!
* [Background_information](#Background_information)
* [Related_Work](#Related_Work)
* [Problem](#Problem)
* [Solution](#Solution)
* [Output](#Output) 
* [Conclusion](#Conclusion) 

<hr>

# Introduction
##  * What Chatbot?
A chatbot is a computer program designed to simulate human conversation and provide automated responses to user inputs. It uses natural language processing (NLP) techniques to understand and interpret user queries, and it generates appropriate responses based on predefined rules, patterns, or machine learning algorithms.
Chatbots can be used in various applications, such as customer support, virtual assistants, information retrieval, and interactive conversational systems.
They can be implemented through different technologies, including rule based systems, retrieval-based models, and generative models.

##  * Why we need Chatbots?
**- Cost and Time Efficiency:** Unlike humans, chatbots can operate 24/7, providing round-the-clock support and instant responses. They can handle multiple conversations simultaneously, ensuring efficient and timely interactions.
**- Affordable Development:** With technological advancements, various development tools haveemerged, making it easier and more cost-effective to create and integrate chatbots. Businesses can invest a relatively small amount and achieve significant benefits.
**- Human-like Interactions:** Modern chatbots are equipped with text-to-speech technology, enabling them to communicate in a more human-like manner. This enhances user experience, as it feels like interacting with a real person on the other side.
**- Business Branding:** Chatbots play a crucial role in modern business strategies, including advertising, branding, and customer engagement. They can promote products and services, provide daily updates, and create a personalized and engaging experience for users.

##  * Type of Chatbots
**- Rule-based chatbots :** operate based on predefined rules and patterns. These chatbots are trained using machine learning models on user intents and corresponding responses. When a user interacts with the chatbot, it analyzes their input and matches it with predefined rules to determine the intent and provide an appropriate response.
**- Self-learning chatbots :** are designed to continuously improve their performance and understanding of user intents without explicit programming. These chatbots utilize advanced technologies such as machine learning, deep learning, and natural language processing (NLP) to learn from user interactions and adapt their responses over time. Self-learning chatbots can autonomously identify user intents and provide more accurate and personalized responses

 <hr>
 
# Background_information
Sentiment analysis is a technique through which you can analyze a piece of text to determine the sentiment behind it. It combines machine learning and natural language processing (NLP) to achieve this.

It is a powerful technique in Artificial intelligence that has important business applications.

For example, you can use sentiment analysis to analyze customer feedback. After collecting that feedback through various mediums like Twitter and Facebook, you can run sentiment analysis algorithms on those text snippets to understand your customers' attitude towards your product.

The primary goal of sentiment analysis is to classify the sentiment of a text as positive, negative, or neutral. However, sentiment analysis can also go beyond simple polarity classification and aim to capture more nuanced emotions, such as joy, anger, sadness, or surprise.

As technology evolves and research progresses, we can expect sentiment analysis chatbots to become more adept at understanding and responding to human emotions, leading to more engaging and effective conversations between humans and AI-powered chatbot systems.

<hr>

# Related_Work
**ChatGPT :** In the beginning of 2023, the world was stunned by the sudden popularity of ChatGPT. The app's mechanics are straightforward enough: type in a prompt at the bottom of the screen and watch as the output appears. Each time the output arrives, a new entry gets added to the left-side menu, which allows one to keep track of all their conversations and return to them whenever they please. Should one conversation strike their fancy, they can share a link and allow others to pick up where they left off.

**Google Bard :** You can now export your input to Google Docs or Gmail drafts. Once you've found the output you want to process, click the "Export" button and choose one of the destinations. The output is fully formatted and ready to use. We hope these are the first of many Google Workspace features coming to Bard. If you're keeping away from Bard because of bad eyes, you can now switch it to dark mode and keep chanting into the wee hours. Compared to ChatGPT, Bard feels more chatty and less focused on text commands.

**HuggingChat:** Introducing HuggingChat, an open source chatbot compiled by Hugging Face. In terms of user experience, it is similar to ChatGPT and also accepts commands Part of the output seems out of place. During one of our conversations, the model told me that some parts of its previous version may have spelling errors. Sometimes, in the middle of the response, no more output is generated. So keep your prompts short.

**Perplexity:** Perplexity is another variation of an AI chatbot that connects to the internet to handle more information and longer, less structured searches. Here's why: when you get the output, you'll see a list of all sources below it. You can then add a new suggestion to continue your search or choose one of the suggested related search terms. All results are stacked below, so you can scroll up or down to read them all. Using Wolfram Alpha, Perplexity is now better equipped for demanding tasks such as solving equations and processing real-time data, and is increasingly positioning itself as a powerful lightweight search tool.

<hr>

# Problem
The main problem that Sentiment Analysis Chatbots aim to address is the need to accurately interpret and respond to the emotional content of user messages. By understanding the sentiment expressed by users, chatbots can provide more personalized and contextually relevant responses. This can be particularly useful in various applications such as customer support, social media monitoring, market research, and content analysis.

By leveraging sentiment analysis, chatbots can offer tailored responses to users based on their emotional state. For example, if a user expresses frustration, the chatbot can provide empathetic and supportive messages. On the other hand, if a user is happy or satisfied, the chatbot can respond with positive reinforcement or additional recommendations.

<hr>

# Solution
To solve the previous problem, we implemented a Chatbot using the Python NLTK library to do sentiment analysis.
## 1- import the library and classes we need to use
    
      import nltk
      from nltk.chat.util import Chat, reflections
      
Chat – Chat is a class that contains complete logic for processing the text data which the chatbot receives and find useful information out of it.
reflections – Another import we have done is reflections which is a dictionary containing basic input and corresponding outputs. You can also create your own dictionary with more responses you want. if you print reflections it will be something like this

## 2- building logic for the NLTK chatbot.
### A. reflections Solution
    
      emotions = {
      ,['سعيد': ]'جميل','حلو','نجاح','تخرج','مبسوط','فرحان','سعيد', 'مبتهج', 'مرتاح'
      ,['حزين': ]'كذب','مهموم','نادم','تحطم','موت','مات','حزين', 'مكسور القلب', 'مكروب'
      ,['غاضب': ]'بغضاء','كره','مستاء','ساخط','عدو','ضرب','غاضب', 'زعالن', 'مستاء'
      ,['متحمس': ]'متحمس','ترقب','فوز','سأفوز','تحدي','مستعد','متفائل','مشوق', 'متلهف'
      ,['خائف': ]'خائف','رهبة','رعب' ,'فزع','قلق','هلع','خشية','مرعوب', 'خوفان'
      ,['مكتئب': ]'جامعة','اختبار','مشروع' ,'اكتئاب','ضغط','البكاء','ضعف','فشل', 'عزله'
      }
      
### B. Generating conversation patterns and responses based on a dictionary of emotions.
      
      conversations = []
      for emotion, words in emotions.items():
      patterns = [r'\b{}\b'.format(word) for word in words]
      [(emotion(format."لديك سبب للشعور بـ}{؟" ,(emotion(format. {}!"أنا أيًضا"] = responses
      conversation = [(pattern, response) for pattern in patterns for response in responses]
      conversations.extend(conversation)

### C- Rules
we have to create rules. The lines of code given below create a simple set of rules. the first line describes the user input which we have taken as raw string input and the next line is our chatbot response.
       
         conversations.extend([ (
         ,"سالم|السالم|مرحًبا|مرحبا|أهال"r
         ["مرحًبا، كيف يمكنني مساعدتك؟", "أهًال، ما الذي تحتاجه؟"]
         ), (
         ,"إىل اللقاء|مع السالمه|مع السالمة"r
         [".أتمنى لك يوًما سعيًدا!", "إىل اللقاء. تفضل بطلبك إذا احتجت مساعدة أخرى"]
         ), (
         , "(*.) اسمي"r
         ["مرحًبا ،1% كيف حالك اليوم؟"]
         ), (
         ,"ما اسمك؟"r
         ["!أنا روبوت تحليل المشاعر تم إنشاؤه بواسطة فاطمة الزهراني.و غدي "]
         ), (
         ,"كيف حالك؟"r
         ["أنا بخير، ماذا عنك؟"]
         ), (
         ,"(*.) آسف"r
         ["ال بأس"]
         ), (
         ,"أنا بخير"r
         ["من الرائع سماع ذلك، كيف يمكنني مساعدتك؟"]
         ), (
         ,"أنا ).*( أفعل الخير"r
         ["(: سررت بسماع ذلك" , "كيف يمكنني مساعدتك؟"]
         ), (
         ,"كم عمرك؟"r
         ["أنا برنامج كمبيوتر، بجدية هل تسألني هذا؟"]
         ), (
         ,"(*.) تم إنشاؤك؟"r
         ["NLTK Python فاطمة وغدي أنشأوني باستخدام مكتبة"]
         ), (
         ,"الموقع | المدينة(؟) (*.)"r
         ["الباحة"]
         ), (
         , "(*.) ما حالة الطقس في"r
         الطقس في 1% رائع كالعادة" , "حار جًدا هنا في 1%" , "بارد جًدا هنا في 1%" ,"لم أسمع حتى"]
         ["عن 1%
         ), (
         ,"(*.) أنا أعمل في"r
         [".شركة رائعة، لقد سمعت عنها. لكنهم يواجهون خسارة كبيرة هذه األيام %1"]
         ), (
         ,"(*.) كيف ).*( الصحة"r
         ["أنا برنامج كمبيوتر، لذا فأنا دائًما بصحة جيدة"]
         ), (
         , "أنا أبحث عن أدلة ودورات عبر اإلنترنت لتعلم علوم البيانات، هل يمكنك اقتراح ذلك؟"r
         عىل العديد من المقاالت الرائعة مع شرح كل خطوة باإلضافة إىل Tech_Crazy يحتوي"]
         ["التعليمات البرمجية، يمكنك استكشافها
         ),
         ,"إنهاء"r( 
         ["(: اعتني بنفسك. أراك قريًبا"]
         ), (
         r".*",
         ["عذًرا، لم أتمكن من فهم ذلك. يمكنك إعادة المحاولة؟"]
         ),])
        
## 3- creates an instance of a chatbot using the Chat class
       
       chatbot = Chat(conversations, reflections)
       
***conversations*** is the list of conversation patterns and responses that define the behavior of the chatbot. These conversation pairs are generated in prev page based on different emotions or predefined patterns.
***reflections*** – Another import we have done is reflections which is a dictionary containing basic input and corresponding outputs. You can also create your own dictionary with more responses you want. if you print reflections it will be something like this.
***Chat*** class is a part of the chatbot implementation and provides methods for processing user input,matching patterns, and generating appropriate responses.

## 4- Create a analyze_emotion function
function defines sentiment analysis or emotion analysis on a given sentence
      
      def analyze_emotion(sentence):
        tokens = nltk.word_tokenize(sentence)
        emotion_scores = {emotion: 0 for emotion in emotions}
        for token in tokens:
          for emotion, words in emotions.items():
            if token in words:
              emotion_scores[emotion] += 1
            max_emotion = max(emotion_scores, key=emotion_scores.get)
        return max_emotion
        
## 5- A function to run the chatbot, and print the feelings after analysis and the appropriate response from the conversation and appropriate words for his current feelings
        
          def run_chatbot():
          السالم عليكم ورحمة هللا , أنا بوت لتحليل المشاعر. اكتب 'مع السالمة'")print
          ("للخروج
          while True:
          (" :أكتب ماذا تحس اآلن ألقوم بتحليل مشاعرك")input = input_user
          if user_input == "السالمة مع "or user_input == "السالمه مع ":
          ("أتمنى لك يوًما سعيًدا مع السالمة (: ")print
          break
          emotion = analyze_emotion(user_input)
          chatbot_response = chatbot.respond(user_input)
          print("مشاعرك:"{} .format(emotion))
          عذًرا، لم أتمكن من فهم ذلك. يمكنك إعادة'=!response_chatbot if
          :'المحاولة؟
          print("الرد:"{} .format(chatbot_response))
          if emotion=="خائف":
              forYou="فَلَا تَخَافُوهُمْ وَخَافُونِ إِن كُنتُم مُّؤْمِنِينَ"
          elif emotion=="غاضب":
              forYou="فَاصْبِرْ لِحُكْمِ رَبِّكَ وَلَا تَكُن كَصَاحِبِ الْحُوتِ إِذْ نَادَىٰ وَهُوَ مَكْظُومٌ"
          elif emotion=="حزين":
              forYou="وَلَا يَحْزُنْكَ قَوْلُهُمْ إِنَّ الْعِزَّةَ لِلَّهِ جَمِيعًا هُوَ السَّمِيعُ الْعَلِيمُ"
          elif emotion=="متحمس":
              forYou="يسْتَبْشِرُونَ بِنِعْمَةٍ مِنَ اللَّهِ وَفَضْلٍ وَأَنَّ اللَّهَ لَا يُضِيعُ أَجْرَ الْمُؤْمِنِينَ"
          elif emotion=="مكتئب":
              forYou=" الّذِينَ آمَنُواْ وَتَطْمَئِنّ قُلُوبُهُمْ بِذِكْرِ اللّهِ أَلاَ بِذِكْرِ اللّهِ تَطْمَئِنّ الْقُلُوبُ"
          else:
              forYou=""
              # forYou="قُلْ بِفَضْلِ اللَّهِ وَبِرَحْمَتِهِ فَبِذَلِكَ فَلْيَفْرَحُوا هُوَ خَيْرٌ مِمَّا يَجْمَعُونَ"
          print(forYou)
    
## 6- Call the previous function to start the conversation
      
      run_chatbot()
      
<hr>

# Output

<img width="500" alt="nlp" src="https://github.com/FatimaALzahrani/Sentiment-analysis-Chabot/assets/107775566/524e5f5a-e9b4-4b24-a384-24289c1c3426">

<hr>


# Conclusion

The project began with an introduction to sentiment analysis, highlighting its significance in understanding and classifying sentiments expressed in text. The background information provided insights into the relevance of sentiment analysis in various domains and its potential to enhance user experiences and decision-making processes.
A review of existing literature and related work in chatbots was conducted.
The project identified the problem of analyzing sentiments in textual data and the need for an automated solution. 
The project proposed the development of a Sentiment Analysis Chatbot tool that combines NLP techniques with sentiment analysis algorithms. 

In conclusion, the project successfully addressed the problem of sentiment analysis in textual data by developing a Sentiment Analysis Chatbot tool. The tool leveraged NLP techniques and sentiment analysis algorithms to provide automated sentiment classification. The project's solution offered an efficient and scalable approach to sentiment analysis, enabling businesses and individuals to gain valuable insights from textual data in real-time.

