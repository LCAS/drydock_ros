#!/usr/bin/env python3


from scene_analyser.scene_analyser_server import SceneAnalyserActionServer


if __name__ == '__main__':

    print( 'starting scene analyser action server' )
    act_server = SceneAnalyserActionServer()
    act_server.spin()
    print( 'scene analyser action server has shut down' )

