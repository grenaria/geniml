library neural_net;

import 'package:bit_array/bit_array.dart';
import 'package:logging/logging.dart';
import 'dart:math';

final _random = Random();

final log = Logger('neural_net');

int _nextRandom(int min, int max) => min + _random.nextInt(max - min);

enum ConnectionType {
  sensor,
  neuron,
  action,
}

enum GeneType {
  neuralConnection,
}

class Gene {
  final GeneType _type = GeneType.neuralConnection;
  late final BitArray _encoding;

  Gene.random() {
    _encoding = BitArray.parseBinary(_random.nextInt(1 << 16).toRadixString(2));
  }

  @override
  String toString() {
    // BitArray test = BitArray.parseBinary('0100110000001111');

    return _encoding.byteBuffer.asInt32List().first.toRadixString(16).padLeft(4, '0');
  }
}

class Genome {
  late final List<Gene> _genes;

  Genome.random(int geneCount) {
    _genes = List.generate(geneCount, (index) => Gene.random());
  }

  @override
  String toString() {
    var buffer = StringBuffer()
      ..writeAll(_genes, ' ');
    return buffer.toString();
  }
}

enum NetworkSensor {
  locX,
  locY,
  boundaryDistX,
  boundaryDist,
  boundaryDistY,
  age,
  random,
}

enum NetworkAction {
  moveX,
  moveY,
  moveRandom,
}

class Neuron {
  double _output = 0;
  double _accumulator = 0;
  bool _driven = false;

  @override
  String toString() {
    return "Internal";
  }
}

class SensorNeuron extends Neuron {
  NetworkSensor _sensor;

  SensorNeuron(this._sensor);

  @override
  String toString() {
    return 'Sensor: ${_sensor.name}';
  }
}

class ActionNeuron extends Neuron {
  NetworkAction _action;

  ActionNeuron(this._action);

  @override
  String toString() {
    return 'Action: ${_action.name}';
  }
}

class NeuralConnection {
  // ConnectionType _sourceType = ConnectionType.sensor;
  // int _sourceNumber = 0; // TODO not sure if this needs to be an enum
  // ConnectionType _sinkType = ConnectionType.action;
  // int _sinkNumber = 0; // TODO same as above

  late final Neuron _source;
  late final Neuron _sink;
  late final int _weight;

  NeuralConnection(this._source, this._sink, this._weight);

  NeuralConnection.randomWeight(this._source, this._sink) {
    _weight = _nextRandom(-32767, 32767);
  }

  double get weight {
    return _weight.toDouble() / 8192.0;
  }

  @override
  String toString() {
    // TODO: implement toString
    return 'Neural Connection: Source[${_source}] Sink[${_sink}] Weight[${_weight}]';
  }
/*
  This is intended to allow genetically derived integers to be used as floats for weights
      static constexpr float f1 = 8.0;
    static constexpr float f2 = 64.0;
    //float weightAsFloat() { return std::pow(weight / f1, 3.0) / f2; }
    float weightAsFloat() const { return weight / 8192.0; }
    static int16_t makeRandomWeight() { return randomUint(0, 0xffff) - 0x8000; }
   */
}

class NeuralNet {
  late List<NeuralConnection> _connections;
  late List<Neuron> _neurons;
  late List<double> _actionLevels;

  NeuralNet.staticTest() {
    _neurons = List<Neuron>.empty(growable: true); // generate(3, (index) => Neuron());
    _neurons.add(SensorNeuron(NetworkSensor.locX));
    _neurons.add(SensorNeuron(NetworkSensor.random));
    _neurons.add(ActionNeuron(NetworkAction.moveX));
    _neurons.add(ActionNeuron(NetworkAction.moveY));

    _connections = List<NeuralConnection>.empty(growable: true);
    _connections.add(NeuralConnection.randomWeight(_neurons[0], _neurons[2]));
    _connections.add(NeuralConnection.randomWeight(_neurons[1], _neurons[3]));

    _actionLevels = List<double>.filled(_connections.length, 0);
  }

  NeuralNet.fromGenome(Genome genome) {
    // TODO write this
    _connections = List<NeuralConnection>.empty();
    _neurons = List<Neuron>.empty();
    _actionLevels = List<double>.empty();
  }

  void feedForward() {
    _actionLevels.map((level) => 0);
    _neurons.map((neuron) => neuron._accumulator = 0);

    // // This container is used to return values for all the action outputs. This array
    // // contains one value per action neuron, which is the sum of all its weighted
    // // input connections. The sum has an arbitrary range. Return by value assumes compiler
    // // return value optimization.
    // std::array<float, Action::NUM_ACTIONS> actionLevels;
    // actionLevels.fill(0.0); // undriven actions default to value 0.0
    //
    // // Weighted inputs to each neuron are summed in neuronAccumulators[]
    // std::vector<float> neuronAccumulators(nnet.neurons.size(), 0.0);
    //
    // // Connections were ordered at birth so that all connections to neurons get
    // // processed here before any connections to actions. As soon as we encounter the
    // // first connection to an action, we'll pass all the neuron input accumulators
    // // through a transfer function and update the neuron outputs in the indiv,
    // // except for undriven neurons which act as bias feeds and don't change. The
    // // transfer function will leave each neuron's output in the range -1.0..1.0.
    //
    // bool neuronOutputsComputed = false;
    // for (Gene & conn : nnet.connections) {
    // if (conn.sinkType == ACTION && !neuronOutputsComputed) {
    // // We've handled all the connections from sensors and now we are about to
    // // start on the connections to the action outputs, so now it's time to
    // // update and latch all the neuron outputs to their proper range (-1.0..1.0)
    // for (unsigned neuronIndex = 0; neuronIndex < nnet.neurons.size(); ++neuronIndex) {
    // if (nnet.neurons[neuronIndex].driven) {
    // nnet.neurons[neuronIndex].output = std::tanh(neuronAccumulators[neuronIndex]);
    // }
    // }
    // neuronOutputsComputed = true;
    // }
    //
    // // Obtain the connection's input value from a sensor neuron or other neuron
    // // The values are summed for now, later passed through a transfer function
    // float inputVal;
    // if (conn.sourceType == SENSOR) {
    // inputVal = getSensor((Sensor)conn.sourceNum, simStep);
    // } else {
    // inputVal = nnet.neurons[conn.sourceNum].output;
    // }
    //
    // // Weight the connection's value and add to neuron accumulator or action accumulator.
    // // The action and neuron accumulators will therefore contain +- float values in
    // // an arbitrary range.
    // if (conn.sinkType == ACTION) {
    // actionLevels[conn.sinkNum] += inputVal * conn.weightAsFloat();
    // } else {
    // neuronAccumulators[conn.sinkNum] += inputVal * conn.weightAsFloat();
    // }
    // }

    // TODO write this
  }

  void executeActions() {
    // TODO write this
  }

  @override
  String toString() {
    var buffer = StringBuffer()
      ..write('Neural Network\n')
      ..writeAll(_connections, '\n');
    return buffer.toString();
  }
}

