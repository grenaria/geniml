library neural_net;

import 'package:bit_array/bit_array.dart';
import 'package:logging/logging.dart';
import 'dart:math';
import 'package:dart_numerics/dart_numerics.dart';

import 'being.dart';

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

abstract class Neuron {
  double _output = 0;
  double _accumulator = 0;
  bool _driven = false;

  @override
  String toString() {
    return "Untyped";
  }

  void _reset() {
    _accumulator = 0;
    _output = 0;
  }
}

class InternalNeuron extends Neuron {
  // TODO write this
  final List<NeuralConnection> _internalConnections = List.empty(growable: true);

  void _addInternalConnection(InternalNeuron neuron, int weight) => _internalConnections.add(NeuralConnection(this, neuron, weight));

  @override
  String toString() {
    return "Internal";
  }
}

class SensorNeuron extends Neuron {
  final NetworkSensor _sensor;
  final List<NeuralConnection> _internalConnections = List.empty(growable: true);
  final List<NeuralConnection> _actionConnections = List.empty(growable: true);

  SensorNeuron(this._sensor);

  void _addActionConnection(ActionNeuron neuron, [int? weight]) => _actionConnections.add(NeuralConnection(this, neuron, weight));
  void _addInternalConnection(InternalNeuron neuron, [int? weight]) => _internalConnections.add(NeuralConnection(this, neuron, weight));

  @override
  String toString() {
    return 'Sensor: ${_sensor.name}';
  }
}

class ActionNeuron extends Neuron {
  final NetworkAction _action;

  ActionNeuron(this._action);

  @override
  String toString() {
    return 'Action: ${_action.name}';
  }
}

class NeuralConnection {
  late final Neuron _source;
  late final Neuron _sink;
  late int _weight;

  NeuralConnection(this._source, this._sink, [int? weight]) {
    _weight = weight ?? _nextRandom(-32767, 32767);
  }

  double get weightAsDouble {
    return _weight.toDouble() / 8192.0;
  }

  @override
  String toString() {
    return 'Neural Connection: Source[$_source] Sink[$_sink] Weight[$_weight]';
  }
}

class NeuralNet {
  late Colony _colony;
  late Being _being;
  final List<SensorNeuron> _sensorNeurons = List.empty(growable: true);
  final List<List<Neuron>> _internalLayers = List.empty(growable: true);
  final List<ActionNeuron> _actionNeurons = List.empty(growable: true);

  // late List<NeuralConnection> _connections;
  // late List<Neuron> _neurons;
  // late List<double> _actionLevels;

  NeuralNet.staticTest(this._colony, this._being) {
    _sensorNeurons.add(SensorNeuron(NetworkSensor.locX));
    _sensorNeurons.add(SensorNeuron(NetworkSensor.random));
    _actionNeurons.add(ActionNeuron(NetworkAction.moveX));
    _actionNeurons.add(ActionNeuron(NetworkAction.moveY));

    _sensorNeurons[0]._addActionConnection(_actionNeurons[0], _randomWeight());
    _sensorNeurons[1]._addActionConnection(_actionNeurons[1], _randomWeight());

    //
    // _connections = List<NeuralConnection>.empty(growable: true);
    // _connections.add(NeuralConnection.randomWeight(_neurons[0], _neurons[2]));
    // _connections.add(NeuralConnection.randomWeight(_neurons[1], _neurons[3]));
    //
    // // TODO need to post-process network. Cull useless neurons, mark undriven/driven neurons based on if they have inputs
    //
    // _actionLevels = List<double>.filled(_connections.length, 0);
  }

  NeuralNet.fromGenome(this._colony, this._being) {
    // TODO write this

    // TODO need to post-process network. Cull useless neurons, mark undriven/driven neurons based on if they have inputs
  }

  void feedForward() {
    // Reset all neurons
    for (var neuron in _actionNeurons) { neuron._reset(); }
    for (var layer in _internalLayers) {
      for (var neuron in layer) { neuron._reset(); }
    }
    for (var neuron in _sensorNeurons) {
      neuron._reset();
      // For sensor neuron layer calculate source strength
      neuron._output = _colony.getEnvironmentSensor(_being, neuron._sensor);
      // Pass it to connections
      for (var connection in neuron._internalConnections) {
        connection._sink._accumulator += (neuron._output * connection.weightAsDouble);
      }
      for (var connection in neuron._actionConnections) {
        connection._sink._accumulator += (neuron._output * connection.weightAsDouble);
      }
    }

    // For internal neuron layers same as sensor neurons except source strength is either bias for undriven neurons or weighted source connections
    // TODO write this



    // For action neurons previous layer calculations should have set action potentials, so set their outputs
    for (var neuron in _actionNeurons) {
      neuron._output = tanh(neuron._accumulator);
    }
  }

  int _randomWeight() => _nextRandom(-32767, 32767);

  void executeActions() {

    // For movement, a number of action neurons control the actual movement. We sum up all of the outputs, apply an activation function to return a probability of movement, then convert the probability to a boolean.

    double moveX = 0;
    double moveY = 0;

    for (var neuron in _actionNeurons) {
      switch (neuron._action) {
        case NetworkAction.moveX:
          moveX += neuron._output;
          break;
        case NetworkAction.moveY:
          moveY += neuron._output;
          break;
        case NetworkAction.moveRandom:
          switch (_nextRandom(0, 3)) {
            case 0:
              moveX += neuron._output;
              break;
            case 1:
              moveX -= neuron._output;
              break;
            case 2:
              moveY += neuron._output;
              break;
            case 3:
              moveY -= neuron._output;
              break;
          }
          break;
      }
    }
    moveX = tanh(moveX);
    moveY = tanh(moveY);

    if (moveX.abs() > _random.nextDouble()) {
      _being.moveX(moveX < 0 ? -1 : 1);
    }

    if (moveY.abs() > _random.nextDouble()) {
      _being.moveY(moveY < 0 ? -1 : 1);
    }
  }

  @override
  String toString() {
    var buffer = StringBuffer()
      ..write('Neural Network\n\nSensor Neurons\n')
      ..writeAll(_sensorNeurons, '\n')
      // TODO write out internal neurons
      ..write('\nActionNeurons\n')
      ..writeAll(_actionNeurons, '\n');
    return buffer.toString();
  }
}

