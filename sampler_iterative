#!/Users/xiaotingzhong/anaconda2/bin/python

import numpy as np
import networkx as nx
# from scipy.optimize import minimize
from copy import deepcopy
import time
import sys
from sys import argv


class Constraint():
	"""Constraints loaded from a file."""

	def __init__(self, fname):
		"""
		Construct a Constraint object from a constraints file

		:param fname: Name of the file to read the Constraint from (string)
		"""
		with open(fname, "r") as f:
			lines = f.readlines()
		# Parse the dimension from the first line
		self.n_dim = int(lines[0])
		# Parse the example from the second line
		self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

		# Run through the rest of the lines and compile the constraints
		self.exprs = []
		for i in range(2, len(lines)):
			# support comments in the first line
			if lines[i][0] == "#":
				continue
			self.exprs.append(compile(lines[i], "<string>", "eval"))
		
		# see the related functions for the variable meaning
		[self.var_constrs, id_vars, self.related_vars] = self.get_var_constraints(lines)
		self.var_groups = self.get_var_groups(id_vars)
		self.bounds = self.get_bounds()
		
		return

	def get_example(self):
		"""Get the example feasible vector"""
		return self.example

	def get_ndim(self):
		"""Get the dimension of the space on which the constraints are defined"""
		return self.n_dim

	def apply(self, x):
		"""
		Apply the constraints to a vector, returning True only if all are satisfied

		:param x: list or array on which to evaluate the constraints
		"""
		for expr in self.exprs:
			if not eval(expr):
				return False
		return True   
	
	def get_var_constraints(self, lines):
		"""
		Re-organize the constraints into a list var_constrs = [n_dim, 2], and its related variables 
		
		:param  vars_inline = [num_constr, ], num_constr is the number of constraints. 
					vars_inline [j]: the variables evaluated in constraint_j. Will be used in self.get_var_groups()
				var_constrs = [n_dim, 2, None], [i, 1, :] contain the linear constraints in which x_i have appeared
					[i, 2, :] contains the non-linear constraints.
				related_vars = [n_dim, 2, None], same structure as var_constrs, records not the contraint expression but
					the varibles that have appeared in var_constrs[i, k, l]. 
					Will be used in get_bounds() and update_current_bounds().
		"""
		vars_inline = []
		id_vars = []

		idx_exprs = 0
		for i in range(2, len(lines)):
				# support comments in the first line
				if lines[i][0] == "#":
					continue
				# record variables involved in this constraint
				id_vars_tmp = []
				for pos, char in enumerate(lines[i]):
					if char == '[':
						interval = lines[i][pos:].find(']')
						id_vars_tmp.append(int(lines[i][pos+1 : pos+interval]))
				vars_inline.append(i)
				# append unique variable ids only
				id_vars.append(list(set(id_vars_tmp)))

		# re-organize the constraints according to the envolved variables
		var_constrs = []
		related_vars = []
		for i in range(self.n_dim):
			var_constrs.append([[], []])
			related_vars.append([[], []])
			
		for i in range(len(vars_inline)):
			idx_linearity = 0
			# check if the constraint is linear
			if any(operator in lines[vars_inline[i]] for operator in ['*', '/', '**']):
				idx_linearity = 1
			# append a contraint to every involved variable
			for j in range(len(id_vars[i])):
				var_constrs[id_vars[i][j]][idx_linearity].append(lines[vars_inline[i]])
				related_vars[id_vars[i][j]][idx_linearity].append(id_vars[i])
		return var_constrs, id_vars, related_vars

	def get_var_groups(self, id_vars):
		"""
		From id_vars returned by get_var_constraints(), construct a graph of the related variables 
		
		:param  id_vars = [num_constr, ], see get_var_constraints() for definition
				
		"""
		G = nx.Graph()
		# construct a variable graph and connect variables appeared in the same constraint
		for i in range(len(id_vars)):
			if len(id_vars[i]) == 0:
				continue 
			elif len(id_vars[i]) == 1:
				G.add_node(id_vars[i][0])
			else:
				idx = 0
				while idx + 1 < len(id_vars[i]):
					G.add_edge(id_vars[i][idx], id_vars[i][idx+1])
					idx += 1
		# get independent components of the variable graph
		var_groups = list(nx.connected_components(G))       
		var_groups = [list(i) for i in var_groups]
		return var_groups
	
	def get_bounds(self):
		"""
		Get variable bounds from linear constraints of this variable.
		Assumption: in the purely linear constraints, every x_i will apprear only once.
		
		:param  bounds_current = [n_dim, 2], low and high bounds of the variables. 
		"""
		bounds = np.zeros([self.n_dim, 2])
		bounds[:,1] = 1.0
		for i in range(self.n_dim):
			# if no linear constraint available, skip 
			if len(self.var_constrs[i][0]) == 0:
				continue
			# loop linear constraints of x_i
			for j in range(len(self.var_constrs[i][0])):
				expr = self.var_constrs[i][0][j]
				expr_vars = self.related_vars[i][0][j]
				# if x_i is the only variable in current linear constraint, then bounds[i,:] can be updated
				if len(expr_vars) == 1:
					expr = self.var_constrs[i][0][j]
					# remove the logical part: >= 0
					idx_expr = find_char_pos(expr, '>')
					expr = expr[0:idx_expr]
					# get the sign of x 
					idx_x = find_char_pos(expr, 'x')
					if idx_x == 0:
						x_sign = '+'
					else:
						x_sign = expr[idx_x-2]
					# evaluate the expression to get bounds
					x = np.zeros(self.n_dim)
					tmp = eval(compile(expr, "<string>", "eval"))
					if x_sign == '+':
						if tmp < 0:
							bounds[i,0] = - tmp
						else:
							bounds[i,1] = 1.0 - tmp
					else:
						if tmp > 0:
							bounds[i,1] = tmp
						else:
							print '!!! intrinsic linear bounds evaluation problem !!!'
		return bounds
		
def find_char_pos(s, ch):
	"""
	auxiliary function for finding character position

	:param  s, string
			ch, character
	"""  
	if len(s) == 1:
		idxes = [i for i, letter in enumerate(s) if letter == ch]
	else:
		idxes = [s.find(ch)]
	if len(idxes) == 1:
		idxes = idxes[0]
	return idxes

def check_constr_i(constrs_i, related_vars_i, x):
	"""
	auxiliary function for checking constraints of one variable. 
	use in sample_data(), see there for detailed information

	:param  constrs_i = constrs[i]
			related_vars_i = related_vars[i]
			info = [bool_passed, fail_idx_1, fail_idx_2]
	"""  
	info = [True, 0, 0]
	# linear constraints
	for j in range(len(constrs_i[0])):
		if len(constrs_i[0][j]) > 1:
			expr = constrs_i[0][j]
			if not eval(compile(expr, "<string>", "eval")):
				info = [False, 0, j]
				return info
	 # non-linear constraints
	for j in range(len(constrs_i[1])):
		expr = constrs_i[1][j]
		if not eval(compile(expr, "<string>", "eval")):
			info = [False, 1, j]
			return info   
	
	return info

def sample_data(constraint, var_idx):
	"""
	sample a group of variables. During the procedure, the feasible region is checked.

	:param  constraint, a class made with Constraint()
			var_idx = [m, ], indexes of the variables to sample
			x__group = [m, ], the sampled variable values
	"""  
	# prepare variables
	bounds = deepcopy(constraint.bounds)
	related_vars = deepcopy(constraint.related_vars)
	constrs = deepcopy(constraint.var_constrs)
	bounds = np.array(bounds)
	x_group = np.random.uniform(low=0.0, high=1.0, size=(constraint.n_dim,))


	idx_tosample = deepcopy(var_idx)
	while len(idx_tosample) > 0:
		# chose a x and update idx_tosample 
		idx = np.random.choice(idx_tosample)
		idx_tosample = np.delete(idx_tosample, np.argwhere(idx_tosample==idx))
		good_data = True

		# check linear constraints of the sampled x[idx], for feasibility and bounds 
		if len(related_vars[idx][0]) > 0:
			for i in range(len(related_vars[idx][0])):
				related_vars_tmp = related_vars[idx][0][i]
				# if the linear constraint doens't only contain x_idx itself 
				if len(related_vars_tmp) > 1:
					idx_sampled = np.setdiff1d(var_idx, idx_tosample)
					# if x_idx is the only unknown variable in this linear constraint, go check 
					if all(np.isin(np.array(related_vars_tmp), idx_sampled)):
						# get possible bound of x
						x = x_group
						x[idx] = 0
						constr_expr = constrs[idx][0][i]
						idx_end = find_char_pos(constr_expr, '>')
						func_expr = constr_expr[:idx_end]
						tmp = eval(compile(func_expr, "<string>", "eval"))
						# get sign of x
						idx_x = find_char_pos(constr_expr, 'x[' + str(idx) + ']')
						if idx_x == 0:
							x_sign = '+'
						else:
							x_sign = constr_expr[idx_x-2]
						# if linear constraint violated, break and resample related variables 
						if (abs(tmp) > 1) or (x_sign == '-' and tmp < 0):
							idx_tosample = np.unique(np.hstack([idx_tosample, related_vars_tmp]))
							good_data = False
							break
						# if linear constraint good, update var_bounds[idx,:]  
						else:
							# update linear bounds
							if x_sign == '+':
								if tmp < 0:
									if - tmp > bounds[idx,0]:
										bounds[idx,0] = - tmp   
								else:
									if 1.0 - tmp < bounds[idx,1]:
										bounds[idx,1] = 1.0 - tmp
							else:
								if tmp > 0:
									if tmp < bounds[idx,1]:
										bounds[idx,1] = tmp
								else:
									print '!!! group bounds evaluation problem !!!'
									print x_group

		# sample a data point 
		x_group[idx] = np.random.uniform(bounds[idx,0], bounds[idx,1],1)
		
#         # check non-linear constraints of the sampled x[idx] 
#         if len(related_vars[idx][1]) > 0:
#             for i in range(len(related_vars[idx][1])):
#                 related_vars_tmp = related_vars[idx][1][i]
#                 idx_sampled = np.setdiff1d(var_idx, idx_tosample)
#                 # if x_idx is the only unknown variable in this constraint, check the constraint # 
#                 if all(np.isin(np.array(related_vars_tmp), idx_sampled)):
#                     [good_constraint, idx_1, idx_2] = check_constr_i(constrs[idx], related_vars[idx], x_group)
#                     if not good_constraint:
#                         idx_tosample = np.unique(np.hstack([idx_tosample, related_vars[idx][idx_1][idx_2]]))
#                         bounds = deepcopy(constraint.bounds)
#                         good_data = False
#                         break 
			
		# after the entire group has been sampled, check all constraints again 
		if len(idx_tosample) == 0:
			for i in var_idx:
				constrs_i = constrs[i]
				related_vars_i = related_vars[i]
				[good_constraint, idx_1, idx_2] = check_constr_i(constrs_i, related_vars_i, x_group)
				if not good_constraint:
					idx_tosample = np.unique(np.hstack([idx_tosample, related_vars_i[idx_1][idx_2]]))
					bounds = deepcopy(constraint.bounds)
					good_data = False
					break 

	return x_group[var_idx]

def main(args):
	start_time = time.time()

	constraint = Constraint(argv[1])
	var_groups = constraint.var_groups
	num_sample = int(argv[3])

	x_list = np.zeros([1000, constraint.n_dim])
	for sample in range(num_sample):
		# Sample variables from graph to graph 
		x = np.array([None] * constraint.n_dim)
		for i in range(len(var_groups)):
			var_idx = var_groups[i]
			x_group = sample_data(constraint, var_idx)
			x[var_idx] = x_group

		if not constraint.apply(x):
			sys.stdout.write("bad condition! \n")
			# print 'bad condition'
			break
			
		x_list[sample, :] = x


	np.savetxt(argv[2], x_list, delimiter=' ', fmt='%1.6f')
	# print("--- %s seconds ---" % (time.time() - start_time))
	sys.stdout.write("--- %s seconds ---\n" % (time.time() - start_time))
	
if __name__ == '__main__':
	if len(argv) < 4:
		print 'Usage: ./sampler [input-file-path] [output-file-path] [param]'
		sys.exit()
	main(argv)