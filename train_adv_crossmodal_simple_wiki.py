import tensorflow as tf
from models.adv_crossmodal_simple_wiki import AdvCrossModalSimple, ModelParams
#from models.wiki_shallow import AdvCrossModalSimple, ModelParams
def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()

    with graph.as_default():
        model = AdvCrossModalSimple(model_params)
    with tf.Session(graph=graph) as sess:
        model.train(sess)
        #model.eval_random_rank()
        model.eval(sess)


if __name__ == '__main__':
    tf.app.run()