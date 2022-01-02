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
  List<Being> _beings = List.empty();

  Colony(int initialBeingCount, int width, int height) {
    log.info('colony initialize');
    _beings = List.generate(initialBeingCount, (int index) {
      return Being(_nextRandom(0, width), _nextRandom(0, height));
    });
  }

  List<Being> get beings {
    return _beings;
  }

  void tick() {
    for (var being in _beings) {
      being.tick();
    }
    log.info('tick');
  }
}

class Being {
  late final Coordinate _location;
  late final BeingSkin _skin;
  late final NeuralNet _brain;
  late final Genome _genome;

  Being(int x, int y) {
    _location = Coordinate(x, y);
    _skin = BeingSkin(_random.nextDouble() * 360);
    _brain = NeuralNet.staticTest();
    _genome = Genome.random(10);

    log.info(_brain);
  }

  void tick() {
    _brain.feedForward();
    _brain.executeActions();
  }

  void moveTo(int x, int y) {
    _location._x = x;
    _location._y = y;
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