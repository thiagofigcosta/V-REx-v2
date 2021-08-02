// SLIDE: https://github.com/keroro824/HashingDeepLearning 

#include "Node.h"

using namespace std;

Node::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, bool useAdamOt, float *weights, float bias, float *adamAvgMom, float *adamAvgVel)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;
    use_adam=useAdamOt;

	if (use_adam)
	{
		_adamAvgMom = adamAvgMom;
		_adamAvgVel = adamAvgVel;
		_t = new float[_dim]();

	}

	_train = new train[_currentBatchsize];
	_activeInputs = 0;

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

}

Node::Node(int dim, int nodeID, int layerID, NodeType type,SlideLabelEncoding labelType, int batchsize, bool useAdamOt, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob){
    use_adam=useAdamOt;
	label_type=labelType;
	_dim = dim;
    _IDinLayer = nodeID;
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    if (use_adam)
    {
        _adamAvgMom = adamAvgMom;
        _adamAvgVel = adamAvgVel;
        _t = new float[_dim]();

    }

    _train = train_blob + nodeID * batchsize;
    _activeInputs = 0;

    _weights = weights;
    _bias = bias;
    _mirrorbias = _bias;
}


Node* Node::createNodeArray(int size, bool useAdamOt,SlideLabelEncoding labelType){
	Node* nodes=new Node[size];
	for (int i=0;i<size;i++){
		nodes[i].use_adam=useAdamOt;
		nodes[i].label_type=labelType;
	}
	return nodes;
}

void Node::Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, float *weights, float bias, float *adamAvgMom, float *adamAvgVel, train* train_blob)
{
    _dim = dim;
    _IDinLayer = nodeID;
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    if (use_adam)
    {
        _adamAvgMom = adamAvgMom;
        _adamAvgVel = adamAvgVel;
        _t = new float[_dim]();

    }

    _train = train_blob + nodeID * batchsize;
    _activeInputs = 0;

    _weights = weights;
    _bias = bias;
    _mirrorbias = _bias;

}

float Node::getLastActivation(int inputID)
{
	if(_train[inputID]._ActiveinputIds != 1)
		return 0.0;
	return _train[inputID]._lastActivations;
}


void Node::incrementDelta(int inputID, float incrementValue)
{
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	#pragma GCC diagnostic pop 
	if (_train[inputID]._lastActivations > 0)
	    _train[inputID]._lastDeltaforBPs += incrementValue;
}

bool Node::getInputActive(int inputID)
{
    return _train[inputID]._ActiveinputIds == 1;
}

bool Node::getActiveInputs(void)
{
    return _activeInputs > 0;
}

float Node::getActivation(int* indices, float* values, int length, int inputID)
{
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));
	#pragma GCC diagnostic pop 
	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
	if (Node::HUGEPAGES){
		if (_train[inputID]._ActiveinputIds != 1) {
			_train[inputID]._ActiveinputIds = 1; //activate input
			_activeInputs++;
		}
	}else{
		_train[inputID]._ActiveinputIds = 1; //activate input
		_activeInputs++;
	}

	_train[inputID]._lastActivations = 0;
	for (int i = 0; i < length; i++)
	{
	    _train[inputID]._lastActivations += _weights[indices[i]] * values[i];
	}
	_train[inputID]._lastActivations += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_train[inputID]._lastActivations < 0) {
		    _train[inputID]._lastActivations = 0;
		    _train[inputID]._lastGradients = 1;
		    _train[inputID]._lastDeltaforBPs = 0;

        }else{
            _train[inputID]._lastGradients = 0;
		}
		break;
	case NodeType::Softmax:
	case NodeType::Sigmoid:
		_train[inputID]._lastActivations = 1/(1+exp(-(_train[inputID]._lastActivations)));
		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

	return _train[inputID]._lastActivations;
}


float Node::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds ==1));
	#pragma GCC diagnostic pop 
	_train[inputID]._lastGradients = 1;
	switch(label_type){
		case SlideLabelEncoding::INT_CLASS:
			_train[inputID]._lastActivations /= normalizationConstant + Slide::SOFTMAX_LINEAR_CONSTANT;
			//TODO:check  gradient
			if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
				_train[inputID]._lastDeltaforBPs = (1.0/labelsize - _train[inputID]._lastActivations) / _currentBatchsize;
			}
			else {
				_train[inputID]._lastDeltaforBPs = (-_train[inputID]._lastActivations) / _currentBatchsize;
			}
			break;
		case SlideLabelEncoding::NEURON_BY_N_LOG_LOSS: {
			//_train[inputID]._lastActivations /= normalizationConstant + Slide::SOFTMAX_LINEAR_CONSTANT; // only for multiples output neurrons
			float a=_train[inputID]._lastActivations;
			float b=1-_train[inputID]._lastActivations;
			if (a==0){
				a=Slide::SOFTMAX_LINEAR_CONSTANT;
			}
			if (b==0){
				b=Slide::SOFTMAX_LINEAR_CONSTANT;
			}
			_train[inputID]._lastDeltaforBPs = -(label[_IDinLayer]*log(a)-(1-label[_IDinLayer])*log(b));
                        // _train[inputID]._lastDeltaforBPs = -(label[_IDinLayer]*log(a)+(1-label[_IDinLayer])*log(b));

			} break;
		case SlideLabelEncoding::NEURON_BY_NEURON:
			 _train[inputID]._lastActivations /= normalizationConstant + Slide::SOFTMAX_LINEAR_CONSTANT;
			_train[inputID]._lastDeltaforBPs = ( label[_IDinLayer] - _train[inputID]._lastActivations ) / (_currentBatchsize*labelsize);
			break;
	}
	string debug="Id: "+to_string(inputID)+" Label idx: "+to_string(_IDinLayer)+" - Neuron: "+to_string(_train[inputID]._lastActivations)+" Expected: "+to_string(label[_IDinLayer])+" Error: "+to_string(_train[inputID]._lastDeltaforBPs)+"\n";
	//cout<<debug;
	return _train[inputID]._lastDeltaforBPs;
}


float Node::backPropagate(node_array previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	float total_grad=0;
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	#pragma GCC diagnostic pop 
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
		Node* prev_node;
		#if Slide_HUGEPAGES == 1
	    	prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
		#else
			prev_node=previousNodes[previousLayerActiveNodeIds[i]];
		#endif
	    prev_node->incrementDelta(inputID, _train[inputID]._lastDeltaforBPs * _weights[previousLayerActiveNodeIds[i]]);

		float grad_t = _train[inputID]._lastDeltaforBPs * prev_node->getLastActivation(inputID);
		total_grad+=abs(_train[inputID]._lastDeltaforBPs);
		if (use_adam)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (use_adam)
	{
		float biasgrad_t = _train[inputID]._lastDeltaforBPs;
		// float biasgrad_tsq = biasgrad_t * biasgrad_t; //unused
		_tbias += biasgrad_t;
	}
	else
    {
        _mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
    }

	_train[inputID]._ActiveinputIds = 0;
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
	_activeInputs--;
	return total_grad/previousLayerActiveNodeSize;
}


float Node::backPropagateFirstLayer(int* nnzindices, float* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	float total_grad=0;
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	#pragma GCC diagnostic pop 
	for (int i = 0; i < nnzSize; i++)
	{
		float grad_t = _train[inputID]._lastDeltaforBPs * nnzvalues[i];
		total_grad+=abs(_train[inputID]._lastDeltaforBPs);
		// float grad_tsq = grad_t * grad_t; // unused
		if (use_adam)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (use_adam)
	{
		float biasgrad_t = _train[inputID]._lastDeltaforBPs;
		//float biasgrad_tsq = biasgrad_t * biasgrad_t; // unused
		_tbias += biasgrad_t;
	}
	else
	{
		_mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
	}

	_train[inputID]._ActiveinputIds = 0;//deactivate inputIDs
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
    _activeInputs--;
	return total_grad/nnzSize;
}

float Node::calcBackPropagateGrad(int previousLayerActiveNodeSize, int inputID)
{
	float total_grad=0;
	#pragma GCC diagnostic push 
    #pragma GCC diagnostic ignored "-Wunused-value"
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	#pragma GCC diagnostic pop 
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		total_grad+=abs(_train[inputID]._lastDeltaforBPs);
	}
	return total_grad/previousLayerActiveNodeSize;
}

void Node::SetlastActivation(int inputID, float realActivation)
{
    _train[inputID]._lastActivations = realActivation;
}

Node::~Node()
{
	delete[] _indicesInTables;
	delete[] _indicesInBuckets;
	if (use_adam)
	{
		delete[] _t;
	}
}


// for debugging gradients.
float Node::purturbWeight(int weightid, float delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


float Node::getGradient(int weightid, int inputID, float InputVal)
{
	return -_train[inputID]._lastDeltaforBPs * InputVal;
}
