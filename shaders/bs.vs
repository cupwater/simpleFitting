#version 130

uniform mat4 MVP;

in vec3 vPos;
in vec3 vColor;

out vec3 color;

void main()
{
	gl_Position = MVP * vec4(vPos, 1.0);
	
	color = vColor;
}