import 'package:flutter/material.dart';
import 'package:logging/logging.dart';
import 'being.dart';
import 'neural_net.dart';
import 'dart:developer' as developer;
import 'dart:async';

const int initialColonySize = 30;
const int colonyWidth = 150;
const int colonyHeight = 150;
const int gridScale = 5;

final log = Logger('main');

scaleInt(int val) {
  return (val * gridScale).toDouble();
}

void main() {
  Logger.root.level = Level.ALL;
  Logger.root.onRecord.listen((record) {
    developer.log(record.message, name: 'geniml.${record.level.name}');
  });

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'geniml',
      theme: ThemeData(
        // primarySwatch: Colors.grey,
        brightness: Brightness.light,
        primaryColor: Colors.teal,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.teal,
        brightness: Brightness.dark,
        primaryColor: Colors.teal,

      ),
      themeMode: ThemeMode.dark,
      debugShowCheckedModeBanner: false,
      home: const MyHomePage(title: 'geniml'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class Tick extends ChangeNotifier {
  late final Timer mainLoop;

  Tick(Colony colony, Duration tickDuration) {
    mainLoop = Timer.periodic(tickDuration, (timer) {
      colony.tick();
      notifyListeners();
    });
  }
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;
  late final Tick tickNotifier;
  final Colony colony = Colony(initialColonySize, colonyWidth, colonyHeight);

  void _incrementCounter() {
    setState(() {
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

  @override
  void initState() {
    super.initState();
    tickNotifier = Tick(colony, const Duration(milliseconds: 10));
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(widget.title),
        backgroundColor: Colors.teal,
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Invoke "debug painting" (press "p" in the console, choose the
          // "Toggle Debug Paint" action from the Flutter Inspector in Android
          // Studio, or the "Toggle Debug Paint" command in Visual Studio Code)
          // to see the wireframe for each widget.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Container(
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.white,
                  ),
                ),
                child: SizedBox(
                    width: scaleInt(colonyWidth) + 1,
                    height: scaleInt(colonyHeight) + 1,
                    child: CustomPaint(
                      painter: ColonyPainter(colony, tickNotifier),
                    )))
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}

class ColonyPainter extends CustomPainter {
  Tick notifier;
  Colony colony;

  ColonyPainter(this.colony, this.notifier) : super(repaint: notifier);

  @override
  void paint(Canvas canvas, Size size) {
    var paint = Paint()
      ..style = PaintingStyle.fill;

    for (var being in colony.beings) {
      paint.color = HSLColor.fromAHSL(1, being.skin.hue, .7, .5).toColor();
      canvas.drawCircle(Offset(scaleInt(being.location.x), scaleInt(being.location.y)), scaleInt(1), paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
