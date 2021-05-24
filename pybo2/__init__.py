from flask import Flask
from flask_cors import CORS


# flask 가상환경 ver : 3.7.3

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .views import main_views
    app.register_blueprint(main_views.bp)

    main_views.call_Model()
    print('call model complete')
    main_views.call_ypr_Model()
    print('call ypr_model complete')
    main_views.call_face_Model()
    print('call face_model complete')
    main_views.call_eye_Model()
    print('call eye_model complete')
    return app






# model = tf.keras.models.load_model("G:/내 드라이브/capstone_2/data/facenet_keras.h5")
# eye_model = load_model('2021_05_19_05_31_31.h5')
# ypr_model = load_model('model.h5')
# face_model = pickle.load(open('G:/내 드라이브/capstone_2/finalized_model.h5', 'rb'))












