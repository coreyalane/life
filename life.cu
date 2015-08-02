#include <GL/glut.h>

#include "game_state.h"
#include "rng.h"

typedef double fp_type;

template<>
fp_type* rng<fp_type>::rand_list = nullptr;
template<>
int rng<fp_type>::rand_index = 0;
template<>
int rng<fp_type>::rand_cycle = 0;

game_state<fp_type>* gs = nullptr;

void draw() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	gs->time_step();
	glutSwapBuffers();
	glFinish();
	glutPostRedisplay();
}

void keyboard(int key, int x, int y) {
	gs->key_presses.emplace(key, x, y);
}

void setupGlut(int argc, char** argv) {
	int WIDTH = 1440;
	int HEIGHT = 720;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowPosition(50, 25);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Life");
	glutDisplayFunc(draw);
	glutSpecialFunc(keyboard);

	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_SMOOTH);        
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);                
        glClearDepth(1);                                       
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glViewport(0,0,WIDTH,HEIGHT);
	glMatrixMode(GL_MODELVIEW);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, 1, -1);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv) {
	cout << "Hello World" << endl;

	cudaSetDevice(0);
	
	rng<fp_type>::init();
	gs = new game_state<fp_type>();
	setupGlut(argc, argv);

	glutMainLoop();

	return 0;
}
