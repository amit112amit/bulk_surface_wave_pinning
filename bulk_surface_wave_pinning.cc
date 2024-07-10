#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/table_handler.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace BSWP {
using namespace dealii;

/**
 * This class implements Bulk-Surface Wave Pinning model.
 */
template <int dim> class WavePinningModel {
public:
  WavePinningModel(std::map<std::string, double> params);
  void run();

private:
  void prepare_grid();
  void setup_system();
  void calculate_F(bool previous_timestep);
  void calculate_G(bool previous_timestep);
  void solve_a_tilde();
  void solve_b();
  void solve_a();
  void output_results(unsigned int, Vector<double> &, Vector<double> &);

  Triangulation<dim> bulk_triangulation;
  Triangulation<dim - 1, dim> surf_triangulation;
  FE_Q<dim> fe_bulk;
  FE_Q<dim - 1, dim> fe_surf;
  DoFHandler<dim> dof_handler_bulk;
  DoFHandler<dim - 1, dim> dof_handler_surf;
  QGauss<dim> quadrature_bulk;
  QGauss<dim - 1> quadrature_surface;
  FEValues<dim> fe_values_bulk;
  FEValues<dim - 1, dim> fe_values_surf;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern_bulk, sparsity_pattern_surface;
  SparseMatrix<double> mass_matrix_bulk, mass_matrix_surface;
  SparseMatrix<double> laplace_matrix_bulk, laplace_matrix_surface;
  SparseMatrix<double> matrix_a, matrix_b;
  SparseMatrix<double> G_a, G_a_old;

  Vector<double> solution_a_tilde, solution_a, solution_b;
  Vector<double> old_solution_a, old_solution_b;
  Vector<double> system_rhs_a, system_rhs_b;
  Vector<double> F_a_b, F_a_b_old;

  SolverControl solver_control;
  SolverCG<Vector<double>> solver;

  TableHandler table_handler;

  std::vector<types::global_dof_index> surf_to_bulk_dof_map;

  double D_a, D_b;
  double time_step;
  double total_time;
  double k0, gamma, K, beta;
};

/**
 * This class is used to set the initial concentration of the membrane bound
 * form.
 */
template <int dim> class InitialValuesA : public Function<dim> {
public:
  InitialValuesA(double al, double ah)
      : Function<dim>(), a_low(al), a_high(ah) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if (dim == 2) {
      // If the point lies beyond the vertical line x=4.75, mark it as high
      if (p(0) > 4.75)
        return a_high;
      else
        return a_low;
    } else if (dim == 3) {
      double R = std::sqrt(p(0) * p(0) + p(1) * p(1));
      if (R < 1.0 && p(2) > 0.0)
        return a_high;
      else
        return a_low;
    } else {
      Assert(false, ExcNotImplemented());
      return 0;
    }
  }

private:
  double a_low, a_high;
};

/**
 * This class will be used to set the initial value of the cytosolic form in the
 * bulk.
 */
template <int dim> class InitialValuesB : public Function<dim> {
public:
  InitialValuesB(double b) : Function<dim>(), b_initial(b) {}
  virtual double value(const Point<dim> &,
                       const unsigned int component = 0) const override {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return b_initial;
  }

private:
  double b_initial;
};

/**
 * We will implement the constructor for the WavePinningModel.
 */
template <int dim>
WavePinningModel<dim>::WavePinningModel(std::map<std::string, double> params)
    : fe_bulk(1), fe_surf(1), dof_handler_bulk(bulk_triangulation),
      dof_handler_surf(surf_triangulation), quadrature_bulk(fe_bulk.degree + 1),
      quadrature_surface(fe_surf.degree + 1),
      fe_values_bulk(fe_bulk, quadrature_bulk,
                     update_values | update_gradients | update_JxW_values),
      fe_values_surf(fe_surf, quadrature_surface,
                     update_values | update_gradients | update_JxW_values),
      solver_control(1000, 1e-12), solver(solver_control) {
  try {
    k0 = params["k0"];
    gamma = params["gamma"];
    K = params["K"];
    beta = params["beta"];
    time_step = params["time_step"];
    total_time = params["total_time"];
    D_a = params["D_a"];
    D_b = params["D_b"];
  } catch (const std::exception &e) {
    std::cout << "KeyError:" << e.what() << std::endl;
  }
}

/**
 * We want to keep the grid preparation separate from the rest of the
 * functionality so that we can import complex grids later on if needed.
 *
 */
template <int dim> void WavePinningModel<dim>::prepare_grid() {
  // For starters, we will create a hyper_ball.
  Point<dim> center;
  for (unsigned int i = 0; i < dim; ++i) {
    center(i) = 0.0;
  }
  GridGenerator::hyper_ball(bulk_triangulation, center, 5.);

  // Let's refine the triangulation a couple of times globally
  bulk_triangulation.refine_global(3);

  // We will identify the cells which are on the boundary and refine them a few
  // times
  for (unsigned int i = 0; i < 2; ++i) {
    for (const auto &cell : bulk_triangulation.active_cell_iterators()) {
      for (const auto &face : cell->face_iterators()) {
        if (face->at_boundary()) {
          cell->set_refine_flag();
        }
      }
    }
    bulk_triangulation.execute_coarsening_and_refinement();
  }

  // Now we will create the surface triangulation from the boundary of the
  // bulk_triangulation
  std::vector<Point<dim>> surf_points;
  std::vector<CellData<dim - 1>> cells;
  for (const auto &cell : bulk_triangulation.active_cell_iterators()) {
    for (const auto &face : cell->face_iterators()) {
      if (face->at_boundary()) {
        CellData<dim - 1> surf_cell;
        for (const auto &vid : face->vertex_indices()) {
          const Point<dim> point = face->vertex(vid);
          types::global_vertex_index vertex_index;
          // Check if this point is already present in the vector
          bool vertex_not_found = true;
          for (unsigned int pid = 0; pid < surf_points.size(); ++pid) {
            if (surf_points[pid].distance(point) < 1e-8) {
              vertex_index = pid;
              vertex_not_found = false;
              break;
            }
          }
          if (vertex_not_found) {
            vertex_index = surf_points.size();
            surf_points.push_back(point);
          }
          surf_cell.vertices[vid] = vertex_index;
        }
        surf_cell.manifold_id = face->manifold_id();
        surf_cell.material_id = cell->material_id();
        cells.push_back(surf_cell);
      }
    }
  }
  // Now we can create the surface triangulation using the points and the cell
  // data
  surf_triangulation.create_triangulation(surf_points, cells, SubCellData());
}

/**
 * In this function, we will calculate the G(a) matrix. Depending on the
 * boolean flag argument, we may use a vector from the previous timestep
 * or the predicted a.
 */
template <int dim>
void WavePinningModel<dim>::calculate_G(bool previous_timestep) {

  // Prepare smartpointer for G and a
  SmartPointer<SparseMatrix<double>> G_ptr;
  SmartPointer<Vector<double>> a_ptr;
  if (previous_timestep) {
    G_ptr = &G_a_old;
    a_ptr = &old_solution_a;
  } else {
    G_ptr = &G_a;
    a_ptr = &solution_a_tilde;
  }

  // Reset G to be all zeros
  (*G_ptr) = 0;

  // Prepare some containers for storing calculated values
  std::vector<double> a_values(fe_values_surf.get_quadrature().size());
  FullMatrix<double> G_a_local(fe_surf.dofs_per_cell);
  std::vector<types::global_dof_index> cell_dofs(fe_surf.dofs_per_cell);

  // Now we are ready to calculate the integral
  for (const auto &cell : dof_handler_surf.active_cell_iterators()) {
    // Prepare the fe_values for the current cell
    fe_values_surf.reinit(cell);
    fe_values_surf.get_function_values(*a_ptr, a_values);
    G_a_local = 0;
    // Here we populate the local G_a matrix
    for (const auto &q_index : fe_values_surf.quadrature_point_indices()) {
      double a_val = a_values[q_index];
      double g_a = k0 + ((gamma * a_val * a_val) / (K * K + a_val * a_val));
      for (const auto &i : fe_values_surf.dof_indices()) {
        for (const auto &j : fe_values_surf.dof_indices()) {
          G_a_local(i, j) += (g_a * fe_values_surf.shape_value(i, q_index) *
                              fe_values_surf.shape_value(j, q_index) *
                              fe_values_surf.JxW(q_index));
        }
      }
    }

    // Copy the local contributions to the global matrix
    cell->get_dof_indices(cell_dofs);
    for (const auto &i : fe_values_surf.dof_indices()) {
      for (const auto &j : fe_values_surf.dof_indices()) {
        G_ptr->add(surf_to_bulk_dof_map[cell_dofs[i]],
                   surf_to_bulk_dof_map[cell_dofs[j]], G_a_local(i, j));
      }
    }
  }
}

/**
 * In this function, we will calculate the F(a, b) vector. Depending on the
 * boolean flag argument, we may use a and b vectors from the previous timestep
 * or the predicted a and b.
 */
template <int dim>
void WavePinningModel<dim>::calculate_F(bool previous_timestep) {
  // Prepare a smart pointer to the appropriate F, a, and b
  SmartPointer<Vector<double>> F_ptr;
  SmartPointer<Vector<double>> a_ptr;
  SmartPointer<Vector<double>> b_ptr;
  if (previous_timestep) {
    F_ptr = &F_a_b_old;
    a_ptr = &old_solution_a;
    b_ptr = &old_solution_b;
  } else {
    F_ptr = &F_a_b;
    a_ptr = &solution_a_tilde;
    b_ptr = &solution_b;
  }

  // Let's make the entries of F zero
  *F_ptr = 0;

  // Now we are ready to calculate the integral
  std::vector<double> a_h_values(fe_values_surf.get_quadrature().size());
  std::vector<types::global_dof_index> cell_dofs(fe_surf.dofs_per_cell);
  for (const auto &cell : dof_handler_surf.active_cell_iterators()) {
    // Prepare the fe_values for the current cell.
    fe_values_surf.reinit(cell);
    fe_values_surf.get_function_values(*a_ptr, a_h_values);
    // Fetch the global dof indices of the cell
    cell->get_dof_indices(cell_dofs);
    for (const auto &q_index : fe_values_surf.quadrature_point_indices()) {
      // To calculate the value of b_val we need to use interpolation
      double b_val = 0;
      for (const auto &i : fe_values_surf.dof_indices()) {
        b_val += (*b_ptr)[surf_to_bulk_dof_map[cell_dofs[i]]] *
                 fe_values_surf.shape_value(i, q_index);
      }
      double a_val = a_h_values[q_index];
      double f_a_b =
          (k0 + ((gamma * a_val * a_val) / (K * K + a_val * a_val))) * b_val -
          beta * a_val;
      for (const auto &i : fe_values_surf.dof_indices()) {
        (*F_ptr)[i] += f_a_b * fe_values_surf.shape_value(i, q_index) *
                       fe_values_surf.JxW(q_index);
      }
    }
  }
}

/**
 * In this function we will store the mass matrices and the laplace matrices
 * that will not change with time.
 *
 */
template <int dim> void WavePinningModel<dim>::setup_system() {

  // Distribute the DoFs for both the grids
  dof_handler_bulk.distribute_dofs(fe_bulk);
  dof_handler_surf.distribute_dofs(fe_surf);

  // We need to map the dofs of the surface and the boundary grids
  std::map<types::global_dof_index, Point<dim>> dof_point_map_bulk =
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler_bulk);
  std::map<types::global_dof_index, Point<dim>> dof_point_map_surf =
      DoFTools::map_dofs_to_support_points(MappingQ1<dim - 1, dim>(),
                                           dof_handler_surf);

  // For every DoF in the surface mesh we need to identify the bulk DoF index
  surf_to_bulk_dof_map.resize(dof_handler_surf.n_dofs());
  for (unsigned int i = 0; i < dof_handler_surf.n_dofs(); ++i) {
    Point<dim> surf_point = dof_point_map_surf[i];
    for (unsigned int j = 0; dof_handler_bulk.n_dofs(); ++j) {
      Point<dim> bulk_point = dof_point_map_bulk[j];
      if (surf_point.distance(bulk_point) < 1e-8) {
        // We found the matching point
        surf_to_bulk_dof_map[i] = j;
        break;
      }
    }
  }

  // Calculate some repeatedly used numbers
  auto n_dofs = dof_handler_bulk.n_dofs();
  auto n_surf_dofs = dof_handler_surf.n_dofs();

  // Make hanging nodes constraints
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_bulk, constraints);
  constraints.close();

  // Create sparsity pattern for the bulk matrices
  DynamicSparsityPattern dsp_bulk(n_dofs, n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler_bulk, dsp_bulk, constraints,
                                  true);
  sparsity_pattern_bulk.copy_from(dsp_bulk);

  // Now we can prepare the surface sparsity patterns
  DynamicSparsityPattern dsp_surface(n_surf_dofs, n_surf_dofs);
  DoFTools::make_sparsity_pattern(dof_handler_surf, dsp_surface);
  sparsity_pattern_surface.copy_from(dsp_surface);

  // We will use the dynamic sparsity patterns for the actual matrices
  mass_matrix_bulk.reinit(sparsity_pattern_bulk);
  mass_matrix_surface.reinit(sparsity_pattern_surface);
  laplace_matrix_bulk.reinit(sparsity_pattern_bulk);
  laplace_matrix_surface.reinit(sparsity_pattern_surface);
  matrix_a.reinit(sparsity_pattern_surface);
  matrix_b.reinit(sparsity_pattern_bulk);

  // The matrices needed at every time step
  G_a.reinit(sparsity_pattern_bulk);
  G_a_old.reinit(sparsity_pattern_bulk);
  F_a_b.reinit(n_surf_dofs);
  F_a_b_old.reinit(n_surf_dofs);

  // Let's set the correct sizes for the Vectors of our problem.
  solution_a_tilde.reinit(n_surf_dofs);
  solution_a.reinit(n_surf_dofs);
  solution_b.reinit(n_dofs);
  old_solution_a.reinit(n_surf_dofs);
  old_solution_b.reinit(n_dofs);
  system_rhs_a.reinit(n_surf_dofs);
  system_rhs_b.reinit(n_dofs);

  // Let's make the mass matrices and the laplace matrices
  MatrixCreator::create_mass_matrix(dof_handler_bulk, quadrature_bulk,
                                    mass_matrix_bulk);
  MatrixCreator::create_laplace_matrix(dof_handler_bulk, quadrature_bulk,
                                       laplace_matrix_bulk);

  MatrixCreator::create_mass_matrix(dof_handler_surf, quadrature_surface,
                                    mass_matrix_surface);
  MatrixCreator::create_laplace_matrix(dof_handler_surf, quadrature_surface,
                                       laplace_matrix_surface);
}

/*
 * This function we will predict the values of $\tilde{a}$ using a^{n-1} and
 * b^{n-1}. At this step will we use IMplicit diffusion and EXplicit reaction.
 * Therefore, this is called IMEX method.
 */
template <int dim> void WavePinningModel<dim>::solve_a_tilde() {
  matrix_a = 0;
  matrix_a.add(1, mass_matrix_surface);
  matrix_a.add(time_step * D_a, laplace_matrix_surface);
  // Now we can prepare the system_rhs_a
  system_rhs_a = 0;
  mass_matrix_surface.vmult(system_rhs_a, old_solution_a);

  // Now we need to calculate F(a^{n-1}, b^{n-1}). It is not clear how to use
  // library functions like create_boundary_rhs_vector() for this case
  calculate_F(true);
  system_rhs_a.add(time_step, F_a_b_old);

  // Now we are ready to solve for a_tilde
  solver.solve(matrix_a, solution_a_tilde, system_rhs_a,
               PreconditionIdentity());
}

/**
 * This function solves for b
 */
template <int dim> void WavePinningModel<dim>::solve_b() {
  // We need to calculate both the G's
  calculate_G(true);
  calculate_G(false);

  // Now we are ready to assemble the coefficient matrix of b
  matrix_b = 0;
  matrix_b.add(1, mass_matrix_bulk);
  matrix_b.add(0.5 * time_step * D_b, laplace_matrix_bulk);
  matrix_b.add(0.5 * time_step, G_a);

  // Now we can assemble the right hand side
  SparseMatrix<double> tmp_mat;
  tmp_mat.reinit(sparsity_pattern_bulk);
  tmp_mat = 0;
  tmp_mat.add(1, mass_matrix_bulk);
  tmp_mat.add(-0.5 * time_step * D_b, laplace_matrix_bulk);
  tmp_mat.add(-0.5 * time_step, G_a_old);

  system_rhs_b = 0;
  tmp_mat.vmult(system_rhs_b, old_solution_b);

  Vector<double> tmp_vec2;
  tmp_vec2.reinit(dof_handler_bulk.n_boundary_dofs());
  tmp_vec2 = old_solution_a;
  tmp_vec2 += solution_a_tilde;
  tmp_vec2 *= 0.5 * time_step * beta;

  Vector<double> tmp_vec3;
  tmp_vec3.reinit(dof_handler_bulk.n_boundary_dofs());
  tmp_vec3 = 0;
  mass_matrix_surface.vmult(tmp_vec3, tmp_vec2);

  // We need to add the contribution of the boundary terms to the rhs
  for (unsigned int i = 0; i < surf_to_bulk_dof_map.size(); ++i) {
    system_rhs_b[surf_to_bulk_dof_map[i]] += tmp_vec3[i];
  }

  // We need to condense the system of equations for the hanging nodes.
  constraints.condense(matrix_b, system_rhs_b);

  // Finally, let's solve for b
  solver.solve(matrix_b, solution_b, system_rhs_b, PreconditionIdentity());

  // Now calculate the constrained node values
  constraints.distribute(solution_b);
}

/*
 * In this function we will correct the values of $\tilde{a}$ to give $a^{n}$
 */
template <int dim> void WavePinningModel<dim>::solve_a() {

  matrix_a = 0;
  matrix_a.add(1, mass_matrix_surface);
  matrix_a.add(0.5 * time_step * D_a, laplace_matrix_surface);

  // Now let's make the rhs
  system_rhs_a = 0;

  SparseMatrix<double> tmp_mat;
  tmp_mat.reinit(sparsity_pattern_surface);
  tmp_mat = 0;
  tmp_mat.add(1, mass_matrix_surface);
  tmp_mat.add(-0.5 * time_step * D_a, laplace_matrix_surface);
  tmp_mat.vmult(system_rhs_a, old_solution_a);

  // F_a_b_old was calculated in solve_a_tilde(), we only need F_a_b
  calculate_F(false);

  system_rhs_a.add(0.5 * time_step, F_a_b);
  system_rhs_a.add(0.5 * time_step, F_a_b_old);

  // Now let's solve the system
  solver.solve(matrix_a, solution_a, system_rhs_a, PreconditionIdentity());
}

/**
 * This function will run the simulation over the time-steps
 */
template <int dim> void WavePinningModel<dim>::run() {
  prepare_grid();
  setup_system();

  // Set the initial values of a
  InitialValuesA<dim> initial_a_values(0.20, 2.35);
  VectorTools::interpolate(dof_handler_surf, initial_a_values, old_solution_a);

  // Set the initial values of b
  InitialValuesB<dim> initial_b_values(1.95);
  VectorTools::interpolate(dof_handler_bulk, initial_b_values, old_solution_b);

  output_results(0, old_solution_a, old_solution_b);

  Vector<double> a_old, b_old;
  a_old.reinit(dof_handler_surf.n_dofs());
  b_old.reinit(dof_handler_bulk.n_dofs());

  unsigned int num_steps = std::round(total_time / time_step);
  for (unsigned int i = 1; i <= num_steps; ++i) {
    solve_a_tilde();
    solve_b();
    solve_a();
    output_results(i, solution_a, solution_b);
    // Update the old solutions
    old_solution_b = solution_b;
    old_solution_a = solution_a;
  }

  // Print the table
  std::ofstream table_file("Conservation.txt");
  table_handler.write_text(table_file);
  std::cout << "Simulation completed." << std::endl;
}

/**
 * This function prints the output files. The first argument vector is the
 * membrane form and the other one is the cytosolic form.
 */
template <int dim>
void WavePinningModel<dim>::output_results(unsigned int t, Vector<double> &a,
                                           Vector<double> &b) {
  // We will have to create a vector of the size of n_dofs() to store a.
  Vector<double> a_full(dof_handler_bulk.n_dofs());
  a_full = 0;
  for (unsigned int i = 0; i < surf_to_bulk_dof_map.size(); ++i) {
    a_full[surf_to_bulk_dof_map[i]] = a[i];
  }
  // Let's check conservation
  std::vector<double> a_values(fe_values_surf.get_quadrature().size());
  double a_total = 0.0;
  for (const auto &cell : dof_handler_surf.active_cell_iterators()) {
    // Calculate the value of a
    fe_values_surf.reinit(cell);
    fe_values_surf.get_function_values(a, a_values);
    for (const auto &q_id : fe_values_surf.quadrature_point_indices()) {
      a_total += (a_values[q_id] * fe_values_surf.JxW(q_id));
    }
  }
  std::vector<double> b_values(fe_values_bulk.get_quadrature().size());
  double b_total = 0.0;
  for (const auto &cell : dof_handler_bulk.active_cell_iterators()) {
    // Calculate the value of b
    fe_values_bulk.reinit(cell);
    fe_values_bulk.get_function_values(b, b_values);
    for (const auto &q_id : fe_values_bulk.quadrature_point_indices()) {
      b_total += (b_values[q_id] * fe_values_bulk.JxW(q_id));
    }
  }
  auto min_max_a = std::minmax_element(std::begin(a), std::end(a));
  auto min_max_b = std::minmax_element(std::begin(b), std::end(b));
  table_handler.add_value("Time_step", t);
  table_handler.add_value("a_min", *min_max_a.first);
  table_handler.add_value("a_max", *min_max_a.second);
  table_handler.add_value("b_min", *min_max_b.first);
  table_handler.add_value("b_max", *min_max_b.second);
  table_handler.add_value("a_total", a_total);
  table_handler.add_value("b_total", b_total);
  table_handler.add_value("a+b", a_total + b_total);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_bulk);
  data_out.add_data_vector(b, "cytosolic_form");
  data_out.add_data_vector(a_full, "membrane_form");
  data_out.build_patches();

  std::string filename("Time-");
  filename += std::to_string(t) + ".vtu";
  std::ofstream outputfile(filename);
  data_out.write_vtu(outputfile);
}

} // namespace BSWP

int main() {
  std::map<std::string, double> params;
  params["k0"] = 0.067;
  params["gamma"] = 1.0;
  params["K"] = 1.0;
  params["beta"] = 1.0;
  params["time_step"] = 0.1;
  params["total_time"] = 10.0;
  params["D_a"] = 0.1;
  params["D_b"] = 10.0;
  auto wpmodel = BSWP::WavePinningModel<3>(params);
  wpmodel.run();
  return 0;
}
