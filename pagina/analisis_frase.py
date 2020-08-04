from textblob import TextBlob
import matplotlib.pyplot as plt
texto = input()
a = TextBlob(texto)
b = a.translate(to='en')
c = b.sentiment
print(b)
print(c)
polarity = c.polarity
subjectivity = c.subjectivity

print ("polaridad es : ",polarity)
print ("subjetividadd es : ",subjectivity)
plt.plot(1,polarity, marker="o", color="red")