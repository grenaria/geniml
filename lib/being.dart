library being;

import 'dart:math';
import 'package:geniml/neural_net.dart';
import 'package:logging/logging.dart';

final _random = Random();

final log = Logger('being');

/// https://stackoverflow.com/questions/13318207/how-to-get-a-random-number-from-range-in-dart
/// Generates a positive random integer uniformly distributed on the range
/// from [min], inclusive, to [max], exclusive.
int _nextRandom(int min, int max) => min + _random.nextInt(max - min);

class Colony {
  late final List<Being> _beings;
  final int _width;
  final int _height;

  Colony(int initialBeingCount, this._width, this._height) {
    log.info('colony initialize');
    _beings = List.generate(initialBeingCount, (int index) {
      return Being(this, _nextRandom(0, _width), _nextRandom(0, _height));
    });
  }

  List<Being> get beings {
    return _beings;
  }

  void tick() {
    for (var being in _beings) {
      being.tick();
    }
    // log.info('tick');
  }

  double getEnvironmentSensor(Being being, NetworkSensor sensor) {
    switch (sensor) {
      case NetworkSensor.locX:
        // log.info('locX sensor: ${being.location.x / _width}, ${being.location.x}, $_width');
        return being.location.x / _width;
      case NetworkSensor.locY:
        return being.location.y / _height;
      case NetworkSensor.boundaryDistX:
        return ((_width - 1) - being.location.x) / (_width - 1);
      case NetworkSensor.boundaryDist:
        return min(((_width - 1) - being.location.x) / (_width - 1), ((_height - 1) - being.location.y) / (_height - 1));
      case NetworkSensor.boundaryDistY:
        return ((_height - 1) - being.location.y) / (_height - 1);
      case NetworkSensor.age:
        if (being.age > 10000) return 1;
        return being.age / 10000;
        // TODO: Handle this case.
        break;
      case NetworkSensor.random:
        return _random.nextDouble();
    }
    log.severe('Tried to get invalid sensor value: ${sensor.name}');
    return 0;
  }

  bool isOpenSpace(int x, int y) {
    // TODO handle collisions
    return x > 0 && y > 0 && x < _width && y < _height;
  }
}

class Being {
  late final Coordinate _location;
  late final BeingSkin _skin;
  late final NeuralNet _brain;
  late final Genome genome;
  late final Colony _colony;
  int age = 0;

  Being(this._colony, int x, int y) {
    _location = Coordinate(x, y);
    _skin = BeingSkin(_random.nextDouble() * 360);
    genome = Genome.random(6);
    _brain = NeuralNet.fromGenome(_colony, this, 5);

    log.info(genome);
    log.info(_brain);
  }

  void tick() {
    _brain.feedForward();
    _brain.executeActions();
    age++;
  }

  void moveTo(int x, int y) {
    if (_colony.isOpenSpace(x, y)) {
      _location._x = x;
      _location._y = y;
    }
  }

  void moveX(int steps) {
    moveTo(_location._x + steps, _location._y);
    // TODO make the move functions honor the colony boundaries and check for occupation
  }

  void moveY(int steps) {
    moveTo(_location._x, _location._y + steps);
  }

  Coordinate get location {
    return _location;
  }

  BeingSkin get skin {
    return _skin;
  }
}

class Coordinate {
  int _x = 0;
  int _y = 0;

  Coordinate(this._x, this._y);

  int get x {
    return _x;
  }

  int get y {
    return _y;
  }
}

class BeingSkin {
  final double _hue;

  BeingSkin(this._hue);

  double get hue {
    return _hue;
  }
}