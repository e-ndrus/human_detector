#Adapted from https://github.com/samlopezf/Python-Email/blob/master/send_email.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

email_user = 'your_email'
email_password = 'your_password'
email_send = 'recipient_email'

def sendDetectedImageToEmail():

    subject = 'Human detected'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Human detected'] = subject

    body = 'Check the picture, the following person is detected!'
    msg.attach(MIMEText(body,'plain'))

    filename='human_detected.png'
    attachment  =open(filename,'rb')

    part = MIMEBase('application','octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',"attachment; filename= "+filename)

    msg.attach(part)
    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,email_password)


    server.sendmail(email_user,email_send,text)
    server.quit()