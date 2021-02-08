
#http://ec2-18-237-1-107.us-west-2.compute.amazonaws.com
from flask import Flask,render_template,redirect,make_response,request
import yagmail
from autocomplete import autocomplete
#ac = autocomplete.load('out.txt')
ac = autocomplete(5)
ac.train('trainer.txt','out.txt')
app = Flask(__name__)

@app.route('/')
def main():
    resp = make_response(render_template('home.html'))
    if(request.args.get('login')=='lilly'):
        resp.set_cookie('login','true')
    else:
        resp.set_cookie('login','false')
    return resp
@app.route('/facetime')
def facetime():
    try:
        if(request.cookies.get('login')=='true'):
            #initializing the server connection
            yag = yagmail.SMTP(user='hifromlillysandmeier@gmail.com', password='**********')
            #sending the email
            yag.send(to='ccummins@u.rochester.edu', subject='Facetime!!', contents='Can you FaceTime me?? -Lilly')
            print("Email sent successfully")
            return render_template('sent.html')
        else:
            print("Error, email was not sent")
            return render_template('error.html')
    except:
        print("Error, email was not sent")
        return render_template('error.html')
@app.route('/attention')
def attention():
    try:
        if(request.cookies.get('login')=='true'):
            #initializing the server connection
            yag = yagmail.SMTP(user='hifromlillysandmeier@gmail.com', password='Luv4soccer.1')
            #sending the email
            yag.send(to='ccummins@u.rochester.edu', subject='Attention!!', contents='Attention?? -Lilly')
            print("Email sent successfully")
            return render_template('sent.html')
        else:
            print("Error, email was not sent")
            return render_template('error.html')
    except:
        print("Error, email was not sent")
        return render_template('error.html')
@app.route('/cuddles')
def cuddles():
    try:
        if(request.cookies.get('login')=='true'):
            #initializing the server connection
            yag = yagmail.SMTP(user='hifromlillysandmeier@gmail.com', password='Luv4soccer.1')
            #sending the email
            yag.send(to='ccummins@u.rochester.edu', subject='Facetime!!', contents='Cuddles!!?? -Lilly')
            print("Email sent successfully")
            return render_template('sent.html')
        else:
            print("Error, email was not sent")
            return render_template('error.html')
    except:
        print("Error, email was not sent")
        return render_template('error.html')
@app.route('/reason')
def reason():
    return render_template('reason.html',content=ac.reason())

if __name__ == '__main__':
    app.run(host='0.0.0.0')